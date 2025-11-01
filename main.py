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
neo4j = Neo4j("your_uri", 
              "neo4j", "your_password")

# Adding nodes to the graph
neo4j.add_nodes(chunks)

# Add relation between nodes
neo4j.precedence_relationship()

# Cleanup
# neo4j.delete_all_relations()
# neo4j.delete_all_nodes()