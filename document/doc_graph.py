from neo4j import GraphDatabase
from time import time
import os
from dotenv import load_dotenv
from pathlib import Path
from pydantic import SecretStr
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
        # Initialize Gemini embeddings
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key is None:
            raise ValueError("GOOGLE_API_KEY not set in environment")
        self.embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", 
                                                       api_key=SecretStr(api_key) if api_key is not None else None,
                                                       output_dimensionality=1536,
                                                       task_type="RETRIEVAL_DOCUMENT")
        self.driver.execute_query("""
                                  CREATE CONSTRAINT unique_chunk IF NOT EXISTS 
                                  FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE
                                  """)
        self.driver.execute_query("DROP INDEX chunk_embeddings IF EXISTS")
        self.driver.execute_query("""
                                  CREATE VECTOR INDEX `chunk_embeddings`
                                  FOR (c:Chunk) ON (c.embeddedChunk)
                                  OPTIONS {
                                      indexConfig: {
                                          `vector.dimensions`: 1536,
                                          `vector.similarity_function`: 'cosine'
                                      }
                                  }
                                  """)
        print("Added constraint for chunkId and created a vector index...")

    def add_nodes(self, list_of_chunks: list, list_of_texts: list):
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
   
    def delete_all_nodes(self):
        '''
            Delete all nodes present in your graph.
        '''
        self.driver.execute_query("""
                                  MATCH (n)
                                  DETACH DELETE n
                                  """)
        print("Deleted all nodes...")

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
      
    def delete_all_relations(self):
        '''
            Delete all relations present in your graph.
        '''
        self.driver.execute_query("""
                                  MATCH ()-[r:PRECEDES]->()
                                  DETACH DELETE r
                                  """)
        print("Deleted all relationships...")
  
    def close(self):
        '''
            Close connection to Neo4j.
        '''
        self.driver.close()
        print("Connection to Neo4j closed...")

    def get_node_with_embedding(self):
        '''
            Returns an ordered list of Records.
                result = [Record, Record, ...]
                Record = < n.chunkId=0, n.embeddedChunk=[float] >

            To get values at any index do - 
                node_id, embeddedVector = result[idx].get("n.chunkId"), result[idx].get("n.embeddedChunk")
        '''
        result = self.driver.execute_query("""
                                            MATCH (n:Chunk)
                                            RETURN n.chunkId, n.embeddedChunk
                                            ORDER BY n.chunkId
                                            """)
        return result

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