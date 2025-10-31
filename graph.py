from neo4j import GraphDatabase

class Neo4j:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, 
                                           auth=(user, password))
        self.driver.verify_connectivity()
        print("Connected to Neo4j successfully...")
        
    def execute_query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return True

    def close(self):
        self.driver.close()

if __name__ == "__main__":
    graph = Neo4j("you_neo4j_uri", 
                  "neo4j", "your_password")