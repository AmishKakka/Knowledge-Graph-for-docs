import os
from doc_loader import DocumentLoader
from graph import Neo4j
from langchain_openai import ChatOpenAI
from pprint import pprint
from time import time
print("Imported required files and packages...")


# Loading, splitting and chunking the file
doc_loader = DocumentLoader()
doc_splits = doc_loader.lazy_load_and_split("AttentionPaper.pdf")

# Creating chunks of the document
chunks, list_texts = doc_loader.create_chunks(doc_splits)

# Connecting to Neo4j
neo4j = Neo4j(os.getenv("uri"), 
              "neo4j", os.getenv("neo4j_pass"))

# Adding nodes to the graph
neo4j.add_nodes(chunks, list_texts)

# Add relation between nodes
neo4j.precedence_relationship()

# Get the node id and text embeddings
st = time()
result = neo4j.get_node_with_embedding()
print("Time taken for embedding retrieval : ", time()-st)

# Query the graph
query = "What is the summary of the paper? Explain the Transformer architecture. Can this be applied to text & images? Also, provide references for your answer to each question."
st = time()
result = neo4j.query(q=query,
                    topK=10)
print("Time taken to query : ", time()-st)
# for record in result.records:
#     print(record.data())

llm = ChatOpenAI(model_name="gpt-4", 
                 temperature=0,
                 openai_api_key=os.getenv('openai'))

messages = [
    ("system", """
                Answer the question based on the context provided from a document.
                --------------
                Use Markdown for all formatting. For example, use bolding for key terms with **text**, and use bullet points for lists, but don't 
                mention it in your response. If you include code snippets, use triple backticks to format them properly.

                If the user asks for references, provide them as a list at the end of your response under a 'References' section. 
                You can refer to the sections that need to be referenced by the page number given as the metadata.
                Also, a final note - if you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
                And the length of the response must be less than 700 words."""),
    ("human", query),
    ("assistant", "Context:\n" + 
                "\n".join([record.data()['c.text'] for record in result.records]) 
                + "\n\nAnswer the question based on the above context." 
                + "Metadata for each context chunk is as follows:\n" +
                "\n".join([f"Chunk from page {record.data()['c.page']}" for record in result.records]))
]
response = llm.invoke(messages)
print(response.content)


# Cleanup
neo4j.delete_all_relations()
neo4j.delete_all_nodes()