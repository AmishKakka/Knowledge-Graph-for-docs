from re import M
from typing import List
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
from pathlib import Path
from pydantic import SecretStr
from chat_data_model import MemoryNode, TurnMemoryOutput
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class Neo4j:
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
        self.driver.execute_query("""
                                CREATE VECTOR INDEX chat_memory_index IF NOT EXISTS
                                FOR (n:Topic) ON (n.embedding)
                                OPTIONS { 
                                  indexConfig: {
                                    `vector.dimensions`: 1536,
                                    `vector.similarity_function`: 'cosine'
                                }}
                            """)
        # Initialize Gemini embeddings
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key is None:
            raise ValueError("GOOGLE_API_KEY not set in environment")
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-2",
            api_key=SecretStr(api_key) if api_key is not None else None,
            output_dimensionality=1536
        )
        print("Created a vector index for chat...")

    def embed_nodes(self, nodes: List[MemoryNode]):
        if not nodes:
            return nodes

        embeddable = [n for n in nodes if n.type != "Turn"]
        if not embeddable:
            return nodes

        response = self.embeddings.embed_documents([n.content for n in embeddable])
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
                tx.run("""
                    MERGE (n {id: $id})
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
                tx.run("""
                    MERGE (n {id: $id})
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
                MERGE (a)-[r:{rel.type.value}]->(b)
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
        memory_output.new_nodes = self.embed_nodes(memory_output.new_nodes)

        with self.driver.session() as session:
            session.execute_write(self._write_turn, memory_output)
        print("New nodes and relations added. Removed resolved questions...")

    def view_all_nodes(self):
        result = self.driver.execute_query("""
                                MATCH (n)
                                RETURN n.id, n.type, n.content
                                """)
        return result
    
    def get_open_questions(self):
        open_questions = self.driver.execute_query("""
                                                   MATCH (q)-[r:UNRESOLVED]->()
                                                   RETURN  q.id, q.type, q.content
                                                   """)
        return open_questions
    
    def delete_all_nodes(self):
        '''
            Delete all nodes present in your graph.
        '''
        self.driver.execute_query("""
                                  MATCH (n)
                                  DETACH DELETE n
                                  """)
        print("Deleted all nodes...")
    
    def delete_all_relations(self):
        '''
            Delete all relations present in your graph.
        '''
        self.driver.execute_query("""
                                  MATCH ()-[r:PRECEDES]->()
                                  DETACH DELETE r
                                  """)
        print("Deleted all relationships...")