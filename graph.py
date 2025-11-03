from neo4j import GraphDatabase
from time import time
import os

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
            Create nodes with given chunk data, also add a embedding vector for each prices of text chunk from the document.
            
            For adding the embedding vector, it is done batchwise for speed.
                Change the value of BATCH_SIZE accordingly.
        '''
        BATCH_SIZE = 50
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
        for batch in range(0, len(list_of_chunks), BATCH_SIZE):  
          i = min(len(list_of_chunks), batch+BATCH_SIZE)
          print(batch, i)
          result = self.driver.execute_query("""
                                      CALL genai.vector.encodeBatch($list_of_texts, 
                                      "OpenAI", 
                                      {token : $api_key, 
                                      model : 'text-embedding-ada-002'}) 
                                      YIELD index, vector
                                      WITH index, [v in vector | round(v, 4)] AS roundedVector
                                      MATCH (c:Chunk {chunkId: $batch_start+index})
                                      SET c.embeddedChunk = roundedVector
                                      RETURN c.chunkId
                                      """,
                                      {'list_of_texts' : list_of_texts[batch:i],
                                       'batch_start' : batch,
                                        'api_key' : os.getenv('openai')}
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
        print("Nodes deleted.")

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
            **Arguments: **
                q: Your query in string format.
                topK : The number of top results to be retrieved that are similar to your query.
            Query the graph.

            **Returns the top 5 relevant chunks with their text value, similarity score, and chunkId**
        '''
        result = self.driver.execute_query("""
                                          WITH genai.vector.encode($text_query,
                                          "OpenAI", 
                                          {token : $api_key, 
                                          model : 'text-embedding-ada-002'}) AS vector
                                          WITH vector AS queryVector
                                          CALL db.index.vector.queryNodes('chunk_embeddings', $topK, queryVector)
                                          YIELD node, score
                                          RETURN node.text, score, node.chunkId
                                          ORDER BY score DESC
                                          """,
                                         {'api_key' : os.getenv('openai'), 'text_query' : q, 'topK' : topK}
                                         )
        return result


if __name__ == "__main__":
    graph = Neo4j(os.getenv("uri"), 
                "neo4j", os.getenv("neo4j_pass"))