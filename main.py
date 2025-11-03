import os
from doc_loader import DocumentLoader
from graph import Neo4j
from pprint import pprint
from time import time
print("Imported required files and packages...")


# Loading, splitting and chunking the file
# doc_loader = DocumentLoader()
# doc_splits = doc_loader.lazy_load_and_split("AttentionPaper.pdf")

# # Creating chunks of the document
# chunks, list_texts = doc_loader.create_chunks(doc_splits)

# Connecting to Neo4j
neo4j = Neo4j(os.getenv("uri"), 
              "neo4j", os.getenv("neo4j_pass"))

# Adding nodes to the graph
# neo4j.add_nodes(chunks, list_texts)

# Add relation between nodes
# neo4j.precedence_relationship()

# Get the node id and text embeddings
# st = time()
# result = neo4j.get_node_with_embedding()
# print("Time taken for embedding retrieval : ", time()-st)

# Query the graph
st = time()
result = neo4j.query(q="What is the summary of the paper? Explain the Transformer architecture. Can this be applied to text & images?",
                    topK=10)
print("Time taken to query : ", time()-st)
for r in result.records:
    print(r.data())
# Cleanup
# neo4j.delete_all_relations()
# neo4j.delete_all_nodes()