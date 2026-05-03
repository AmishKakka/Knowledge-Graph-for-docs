from neo4j import GraphDatabase
from time import time
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class Neo4j:
    def __init__(self, uri, user, password):
        '''
            Returns nothing. 
            
            Outputs if the connection is successful.
            Creates a Vector index and constraint for a unique id for each node.
            In our case this will be the chunkId.
        '''
        self.driver = GraphDatabase.driver(uri, 
                                           auth=(user, password))
        self.driver.verify_connectivity()
        print("Connected to Neo4j successfully...")
        # Initialize Google Generative AI embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2", output_dimensionality=1536)
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
        '''
            Create nodes with given chunk data, also add an embedding vector for each text chunk from the document.
            
            Embeddings are generated using Google Generative AI and added batchwise for speed.
                Change the value of BATCH_SIZE accordingly.
        '''
        BATCH_SIZE = 100
        st = time()
        self.driver.execute_query("""
                                  UNWIND $list_of_chunks as chunkParam
                                  MERGE (c: Chunk{
                                      text: chunkParam['text'],
                                      file_path: chunkParam['file_path'],
                                      page: chunkParam['page'],
                                      chunkId: chunkParam['chunkId']
                                  })
                                  """,
                                 {'list_of_chunks': list_of_chunks})
        
        # Generate embeddings in batches using LangChain
        for batch in range(0, len(list_of_texts), BATCH_SIZE):  
            i = min(len(list_of_texts), batch+BATCH_SIZE)
            print(f"Processing batch {batch} to {i}")
            
            # Generate embeddings for this batch
            batch_texts = list_of_texts[batch:i]
            embeddings = self.embeddings.embed_documents(batch_texts)
            
            # Prepare batch data for bulk update
            batch_data = []
            for idx, embedding in enumerate(embeddings):
                chunk_id = batch + idx
                rounded_embedding = [round(v, 4) for v in embedding]
                batch_data.append({
                    'chunkId': chunk_id,
                    'embedding': rounded_embedding
                })
            
            # Bulk update all nodes in this batch with one transaction
            self.driver.execute_query("""
                                      UNWIND $batch_data as item
                                      MATCH (c:Chunk {chunkId: item.chunkId})
                                      SET c.embeddedChunk = item.embedding
                                      RETURN c.chunkId
                                      """,
                                      {'batch_data': batch_data}
                                      )        
        print("Time taken: ", time()-st)
        print("Added nodes with text embeddings to the graph...")

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
    
    def query(self, q: str, topK: int):
        '''
            Query the graph.
            
            **Arguments:**
                q: Your query in string format.
                topK : The number of top results to be retrieved that are similar to your query.

            **Returns the top K relevant chunks with their text value, similarity score, and chunkId**
        '''
        # Generate query embedding using LangChain
        query_embedding = self.embeddings.embed_query(q)
        
        result = self.driver.execute_query("""
                                          WITH $queryVector AS queryVector
                                          CALL db.index.vector.queryNodes('chunk_embeddings', $topK, queryVector)
                                          YIELD node AS c, score
                                          OPTIONAL MATCH (n1:Chunk)-[:PRECEDES]->(c)
                                          OPTIONAL MATCH (c)-[:PRECEDES]->(n2:Chunk)
                                          RETURN n1.text, n2.text, c.text, c.page, n1.page, n2.page, score
                                          ORDER BY score DESC
                                          """,
                                         {'queryVector': query_embedding, 'topK': topK}
                                         )
        return result
