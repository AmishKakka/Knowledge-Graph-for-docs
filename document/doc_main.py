import os
from document.doc_loader import DocumentLoader
from document.doc_graph import Neo4j
from langchain_google_genai import ChatGoogleGenerativeAI
from pprint import pprint
from dotenv import load_dotenv
from pathlib import Path
from time import time
from data_model import GraphOutput
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

# Extract text from Document objects
doc_texts = "\n\n---\n\n".join([f"Page {doc.metadata['page']}, Chunk {doc.metadata.get('chunkId', 'N/A')}:\n{doc.page_content}" for doc in doc_splits[:15]])

prompt = [
    ("system", """
                You are a Research Knowledge Graph specialist. Your goal is to extract a high-fidelity graph from academic text.

                1. **Entity Identification**:
                - Author/Person: Use full names if available.
                - Concept/Idea: Extract core methodologies, theories, or scientific terms.
                - Document: The specific research paper or section title.

                2. **Relationship Predicates (Strictly use these)**:
                - AUTHOR_OF: Person to ResearchPaper.
                - AFFILIATED_WITH: Person to Institution.
                - CONTRIBUTES_TO: ResearchPaper to Concept.
                - RELATED_TO: Concept to Concept.
                - CITES: ResearchPaper to ResearchPaper.

                3. **Normalization**: Ensure entity IDs are canonical. Avoid acronyms unless they are the primary identifier (e.g., use "Recurrent Neural Network" instead of "RNN" for the node ID).

                4. **Provenance**: Link every node and relationship to the provided page and chunkId.
                """),
    ("human", doc_texts)
]

# Structuring the LLM with specific output schema
structured_llm = llm.with_structured_output(GraphOutput)
structured_response = structured_llm.invoke(prompt)
pprint(structured_response)

neo4j.delete_all_relations()
neo4j.delete_all_nodes()