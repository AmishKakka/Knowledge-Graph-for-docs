import os
from doc_loader import DocumentLoader
from graph import Neo4j
from pprint import pprint
print("Imported required files and packages...")


# Loading, splitting and chunking the file
doc_loader = DocumentLoader()
doc_splits = doc_loader.lazy_load_and_split("AttentionPaper.pdf")

# Creating chunks of the document
chunks = doc_loader.create_chunks(doc_splits)

# Connecting to Neo4j
neo4j = Neo4j("neo4j+s://6fa154bf.databases.neo4j.io", 
              "neo4j", "Vs1de404s4YLJP0I4lbuj5lyLaVzdMEpbK5KDIsUyCc")

# Adding nodes to the graph
neo4j.add_nodes(chunks)