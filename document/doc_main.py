import os
from doc_loader import DocumentLoader
from doc_graph import Neo4j
from langchain_google_genai import ChatGoogleGenerativeAI
from pprint import pprint
from dotenv import load_dotenv
from pathlib import Path
from time import time
print("Imported required files and packages...")

load_dotenv(Path(__file__).parent.parent / ".env")
neo4j_pwd = os.getenv("NEO4J_Password")

# Loading, splitting and chunking the file
doc_loader = DocumentLoader()
doc_splits = doc_loader.lazy_load_and_split("AttentionPaper.pdf")

# Creating chunks of the document
chunks, list_texts = doc_loader.create_chunks(doc_splits)

# Connecting to Neo4j
neo4j = Neo4j(
            "neo4j://0.0.0.0:7687",    # Connecting to the Neo4j instance running inside your docker container
            "neo4j", neo4j_pwd
            )

# Adding nodes to the graph
neo4j.add_nodes(chunks, list_texts)

# Add relation between nodes
neo4j.precedence_relationship()


# Query the graph
query = "Are you based on Transformer architecture? Is it better than Diffusion models?"
st = time()
relevant_doc_nodes = neo4j.get_relevant_doc_nodes(query, topK=2)
print("Time taken to query : ", time()-st)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                              temperature=0,
                              google_api_key=os.getenv('GOOGLE_API_KEY'))


# Sending the LLM simple text query and context to generate a repsonse
if relevant_doc_nodes:
    formatted_nodes = "\n".join([
        f"- [{node['page']}] {node['text']} (relevance: {node['score']:.2f})"
        for node in relevant_doc_nodes
    ])
    context_section = f"""
                        ## Retrieved Context
                        The following facts were retrieved, ordered by relevance to the current query:

                        {formatted_nodes}

                        ## Instructions
                        1. PRIORITIZE the retrieved context above when it directly answers the question.
                        2. If the context is relevant but incomplete, supplement with your knowledge 
                            and clearly distinguish: "From our conversation: X. Additionally: Y."
                        3. If the context is irrelevant to the query, ignore it and answer normally.
                        """
else:
    context_section = """
                    ## Retrieved Memory Context
                    No relevant context found.
                    Answer the question using your general knowledge.
                    """
prompt = [
      ("system", f"""
      You are a helpful assistant with graph-based memory.
      You have access to semantically retrieved facts for a query.
      
      {context_section}
      """),
      ("human", query)
]
response = llm.invoke(prompt)
print(response.content)
