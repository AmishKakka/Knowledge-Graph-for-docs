from neo4j import GraphDatabase
from time import time
import os
from dotenv import load_dotenv
from pathlib import Path
from pydantic import SecretStr
from typing import List
from chat_system.chat_data_model import MemoryNode, TurnMemoryOutput
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class Neo4j:
    '''
        Connect to your hosted Neo4j database
    '''
    def __init__(self, uri, user, password):
        '''
            Returns nothing. 
            
            Outputs if the connection is successful.
            Creates a Vector index and constraint for a unique id for each node.
            In our case this will be the chunkId.
        '''
        load_dotenv(Path(__file__).parent.parent / ".env")
        self.driver = GraphDatabase.driver(uri, 
                                           auth=(user, password))
        self.driver.verify_connectivity()
        print("Connected to Neo4j successfully...")
        # Initialize Gemini embeddings
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key is None:
            raise ValueError("GOOGLE_API_KEY not set in environment")
        self.embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", 
                                                       api_key=SecretStr(api_key) if api_key is not None else None,
                                                       output_dimensionality=1536,
                                                       task_type="RETRIEVAL_DOCUMENT")
        
        with self.driver.session() as session:
            with session.begin_transaction() as tx:
                tx.run("""
                        CREATE CONSTRAINT unique_chunk IF NOT EXISTS 
                        FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE
                        """)
                # Dropping previuosly created indexes
                tx.run("DROP INDEX chunk_embeddings IF EXISTS;")
                tx.run("DROP INDEX chat_memory_index IF EXISTS;")
                
                # Creating new vector indexes
                tx.run("""CREATE VECTOR INDEX `chunk_embeddings` IF NOT EXISTS
                            FOR (c:Chunk) ON (c.embeddedChunk)
                            OPTIONS {
                                indexConfig: {
                                    `vector.dimensions`: 1536,
                                    `vector.similarity_function`: 'cosine'
                                }
                            };"""
                       )
                tx.run("""CREATE VECTOR INDEX `chat_memory_index` IF NOT EXISTS
                            FOR (n:Memory) ON (n.embedding)
                            OPTIONS { 
                                indexConfig: {
                                    `vector.dimensions`: 1536,
                                    `vector.similarity_function`: 'cosine'
                                }
                            };"""
                       )
        print("Added constraint for chunkId and created a vector index...")

    def add_chat_nodes(self, nodes: List[MemoryNode]):
        if not nodes:
            return nodes

        embeddable = [n for n in nodes if n.type != "Turn"]
        if not embeddable:
            return nodes

        response = []
        for n in embeddable:
            response.append(self.embeddings.embed_documents([n.content]))
        print(f"Nodes to embed: {len(embeddable)} | Embeddings returned: {len(response)}")
            
        # Attach embeddings back
        embedding_map = { node.id: embedding for node, embedding in zip(embeddable, response) }

        for node in nodes:
            if node.id in embedding_map:
                node.embedding = embedding_map[node.id]
        print("Embeddings for nodes created...")
        return nodes
    
    @staticmethod
    def _write_turn(tx, memory_output: TurnMemoryOutput):
        # Upsert Nodes
        for node in memory_output.new_nodes:
            if node.embedding is not None:
                tx.run(f"""
                    MERGE (n:{node.type}:Memory {{id: $id}})
                    SET n.type = $type,
                        n.content = $content,
                        n.turn_id = $turn_id,
                        n.embedding = $embedding
                    WITH n
                    CALL db.create.setNodeVectorProperty(n, 'embedding', $embedding)
                """,
                id=node.id,
                type=node.type,
                content=node.content,
                turn_id=node.turn_id,
                embedding=node.embedding
                )
            else:
                tx.run(f"""
                    MERGE (n:{node.type} {{id: $id}})
                    SET n.type = $type,
                        n.content = $content,
                        n.turn_id = $turn_id
                """,
                id=node.id,
                type=node.type,
                content=node.content,
                turn_id=node.turn_id,
                )

        # New Relationships
        for rel in memory_output.new_relationships:
            tx.run(f"""
                MATCH (a {{id: $source}})
                MATCH (b {{id: $target}})
                MERGE (a)-[r:{rel.type}]->(b)
                SET r.turn_id = $turn_id
            """,
            source=rel.source,
            target=rel.target,
            turn_id=rel.turn_id
            )

        # Resolve Questions
        for question_id in memory_output.resolved_question_ids:
            tx.run("""
                MATCH (q {id: $question_id})-[r:UNRESOLVED]->()
                DELETE r
                SET q.resolved = true
            """,
            question_id=question_id
            )

    def add_turn(self, memory_output):
        for node in memory_output.new_nodes:
            node.type = node.type.value if hasattr(node.type, 'value') else node.type
        for rel in memory_output.new_relationships:
            rel.type = rel.type.value if hasattr(rel.type, 'value') else rel.type

        memory_output.new_nodes = self.add_chat_nodes(memory_output.new_nodes)

        with self.driver.session() as session:
            session.execute_write(self._write_turn, memory_output)
        print("New nodes and relations added. Removed resolved questions...")
    
    def add_document_nodes(self, list_of_chunks: list, list_of_texts: list):
        BATCH_SIZE = 100
        st = time()

        for batch in range(0, len(list_of_texts), BATCH_SIZE):
            i = min(len(list_of_texts), batch + BATCH_SIZE)
            print(f"Processing batch {batch} to {i}")

            batch_texts  = list_of_texts[batch:i]
            batch_chunks = list_of_chunks[batch:i]

            # Generate embeddings for this batch
            embeddings = self.embeddings.embed_documents(batch_texts)
            print("No. of embedded text chunks: ", len(embeddings))

            # chunk metadata + embedding
            batch_data = [
                {
                    'text': chunk['text'],
                    'file_path': chunk['file_path'],
                    'page': chunk['page'],
                    'chunkId': chunk['chunkId'],
                    'embedding': embedding
                }
                for chunk, embedding in zip(batch_chunks, embeddings)
            ]
            print("Length of current batch after embedding: ", len(batch_data))
            self.driver.execute_query("""
                                    UNWIND $batch_data AS item
                                    MERGE (c:Chunk {chunkId: item.chunkId})
                                    SET c.text = item.text,
                                        c.file_path = item.file_path,
                                        c.page = item.page,
                                        c.embeddedChunk = item.embedding
                                    WITH c, item
                                    CALL db.create.setNodeVectorProperty(c, 'embeddedChunk', item.embedding)
                                """, 
                                {'batch_data': batch_data})
        print("Time taken: ", time() - st)
        print("Added nodes with embeddings to the graph...")   

    def get_relevant_doc_nodes(self, query: str, topK: int = 3):
        '''
            Query the graph.
            
            **Arguments:**
                q: Your query in string format.
                topK : The number of top results to be retrieved that are similar to your query.

            **Returns the topK relevant chunks with their text value, similarity score, and chunkId**
        '''
        query_embedding = self.embeddings.embed_query(query)
        result = self.driver.execute_query("""
                            CALL db.index.vector.queryNodes(
                                'chunk_embeddings', $top_k, $embedding
                            )
                            YIELD node, score
                                        
                            WITH node, score
                            OPTIONAL MATCH (node)-[r]-(neighbour)
                            
                            RETURN 
                                node.id AS chunk_id,
                                node.text AS text,
                                node.page AS page,
                                score,
                                collect(neighbour.text) AS neighbour_contents
                            ORDER BY score DESC
                        """,
                        top_k=topK,
                        embedding=query_embedding
                        )
        relevant_nodes = [record.data() for record in result.records]
        print(f" Nodes relevant to {query}:  {relevant_nodes}")
        return relevant_nodes
    
    def view_all_nodes(self):
        result = self.driver.execute_query("""
                                MATCH (n)
                                RETURN n.id, n.type, n.content
                                """)
        existing_nodes = [res.data() for res in result.records]
        return existing_nodes
    
    def get_open_questions(self):
        result = self.driver.execute_query("""
                                            MATCH (q)-[r:UNRESOLVED]->()
                                            RETURN  q.id, q.type, q.content
                                            """)
        open_questions = [res.data() for res in result.records]
        return open_questions
    
    def get_relevant_chat_nodes(self, query: str, topK: int = 5):
        query_embedding = self.embeddings.embed_query(query)
        result = self.driver.execute_query("""
                            CALL db.index.vector.queryNodes(
                                'chat_memory_index', $top_k, $embedding
                            )
                            YIELD node, score
                                           
                            WITH node, score
                            OPTIONAL MATCH (node)-[r]-(neighbour)
                            
                            RETURN 
                                node.id AS id,
                                node.type AS type,
                                node.content AS content,
                                score,
                                collect(neighbour.content) AS neighbour_contents
                            ORDER BY score DESC
                        """,
                        top_k=topK,
                        embedding=query_embedding
                        )
        relevant_nodes = [record.data() for record in result.records]
        print(f" Nodes relevant to {query}:  {relevant_nodes}")
        return relevant_nodes
    
    def precedence_relationship(self):
        '''
            Add a simple precedence relation based on chunkIDs in an increasing order.
        '''
        self.driver.execute_query("""
                                  MATCH (c1:Chunk)
                                  MATCH (c2:Chunk {chunkId: c1.chunkId + 1})
                                  CREATE (c1)-[r:PRECEDES]->(c2)
                                  """)
        print("Added precedence relations between nodes...")
    
    def delete_all_nodes(self):
        '''
            Delete all nodes present in your graph.
        '''
        self.driver.execute_query("""
                                  MATCH (n)
                                  DETACH DELETE n
                                  """)
        print("Deleted all nodes...")
      
    def delete_precedence_relations(self):
        '''
            Delete all relations present in your graph.
        '''
        self.driver.execute_query("""
                                  MATCH ()-[r:PRECEDES]->()
                                  DETACH DELETE r
                                  """)
        print("Deleted all relationships...")

    def delete_all_relations(self):
        '''
            Delete all relations present in your graph.
        '''
        self.driver.execute_query("""
                                  MATCH ()-[r]->()
                                  DETACH DELETE r
                                  """)
        print("Deleted all relationships...")


# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––– #
# load_dotenv(Path(__file__).parent / ".env")
# neo4j_pwd = os.getenv("NEO4J_Password")
# neo4j = Neo4j(
#             "neo4j://0.0.0.0:7687",    # Connecting to the Neo4j instance running inside your docker container
#             "neo4j", neo4j_pwd
#             )