from neo4j import GraphDatabase
from time import time


class Neo4j:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, 
                                           auth=(user, password))
        self.driver.verify_connectivity()
        print("Connected to Neo4j successfully...")
        self.driver.execute_query("""
                                  CREATE CONSTRAINT unique_chunk IF NOT EXISTS 
                                  FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE
                                  """)
        print("Added constraint for chunkId...")
        
    def add_nodes(self, list_of_chunks: list):
        st = time()
        result = self.driver.execute_query("""
                                            UNWIND $list_of_chunks as chunkParam
                                            MERGE (c: Chunk{
                                                text: chunkParam['text'],
                                                file_path: chunkParam['file_path'],
                                                page: chunkParam['page'],
                                                chunkId: chunkParam['chunkId']
                                            })
                                            RETURN c.chunkId, timestamp()
                                            """,
                                           {'list_of_chunks': list_of_chunks})
        print("Time taken: ", time()-st)
        print("Added nodes to the graph...")
        # print(result)

    def delete_all_nodes(self):
        self.driver.execute_query("""
                                  MATCH (n)
                                  DETACH DELETE n
                                  """)
        print("Nodes deleted.")

    def precedence_relationship(self):
        self.driver.execute_query("""
                                  MATCH (c1:Chunk)
                                  MATCH (c2:Chunk {chunkId: c1.chunkId + 1})
                                  CREATE (c1)-[r:PRECEDES]->(c2)
                                  """)
        print("Added precedence relations between nodes...")
      
    def delete_all_relations(self):
        self.driver.execute_query("""
                                  MATCH ()-[r:PRECEDES]->()
                                  DETACH DELETE r
                                  """)
        print("Deleted all relationships...")
  
    def close(self):
        self.driver.close()
        print("Connection to Neo4j closed...")


if __name__ == "__main__":
    graph = Neo4j("your_uri", 
                "neo4j", "your_password")