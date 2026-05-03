import os
from doc_loader import DocumentLoader
from graph import Neo4j
from langchain_google_genai import ChatGoogleGenerativeAI
from pprint import pprint
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from typing import Sequence
from time import time
print("Imported required files and packages...")


# Loading, splitting and chunking the file
doc_loader = DocumentLoader()
doc_splits = doc_loader.lazy_load_and_split("AttentionPaper.pdf")

# Creating chunks of the document
chunks, list_texts = doc_loader.create_chunks(doc_splits)

# Connecting to Neo4j
neo4j = Neo4j(
            "neo4j://0.0.0.0:7687", 
            "neo4j", "your_password"
            )

# Adding nodes to the graph
neo4j.add_nodes(chunks, list_texts)

# Add relation between nodes
neo4j.precedence_relationship()

# Get the node id and text embeddings
st = time()
result = neo4j.get_node_with_embedding()
print("Time taken for embedding retrieval : ", time()-st)

# Query the graph
query = "What is the summary of the paper? Explain the Transformer architecture. \
Can this be applied to text & images? Also, provide references for your answer to each question."
# query = "What was my last question?"
st = time()
result = neo4j.query(q=query,
                    topK=10)
print("Time taken to query : ", time()-st)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                              temperature=0,
                              google_api_key=os.getenv('GOOGLE_API_KEY'))

# ---------------------------------------------------------- #
# Sending the LLM simple text query and some context to generate a repsonse

# messages = [
#     ("system", """
#                 Answer the question based on the context provided from a document.
#                 --------------
#                 Use Markdown for all formatting. For example, use bolding for key terms with **text**, and use bullet points for lists, but don't 
#                 mention it in your response. If you include code snippets, use triple backticks to format them properly.
#                 If the user asks for references, provide them as a list at the end of your response under a 'References' section. 
#                 You can refer to the sections that need to be referenced by the page number given as the metadata.
#                 Also, a final note - if you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
#                 Be brief in your responses, if asked to answer under 200 words do so, but if nothing is mentioned about the length, provide a concise answer with required elaboration"""),
#     ("human", query),
#     ("assistant", "Context:\n" + 
#                 "\n".join([record.data()['c.text'][:1000] for record in result.records][:7])  # Limit each chunk to 1000 chars and take top 7 chunks
#                 + "\n\nAnswer the question based on the above context." 
#                 + "\nMetadata for each context chunk is as follows:\n" +
#                 "\n".join([f"Chunk from page {record.data()['c.page']}" for record in result.records][:7]))
# ]
# response = llm.invoke(messages)
# print(response.content)


# ---------------------------------------------------------- #
# Invoking the LLM to generate graph documents

st = time()
graphTransformer = LLMGraphTransformer(llm=llm)
# Sending just a single document chunk
graph_documents = graphTransformer.convert_to_graph_documents(doc_splits)
print("Time taken to convert to graph: ", time()-st)

for docs in graph_documents:
    pprint(docs.nodes)
    print("-----")
    pprint(docs.relationships)
    print("-----")
    pprint(docs.source)
    

# Cleanup
neo4j.delete_all_relations()
neo4j.delete_all_nodes()