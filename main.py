import os
from doc_loader import DocumentLoader
from graph import Neo4j
from langchain_openai import ChatOpenAI
from pprint import pprint
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.prompts import ChatPromptTemplate
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
# neo4j.add_nodes(chunks, list_texts)

# Add relation between nodes
# neo4j.precedence_relationship()

# Get the node id and text embeddings
# st = time()
# result = neo4j.get_node_with_embedding()
# print("Time taken for embedding retrieval : ", time()-st)

# Query the graph
# query = "What is the summary of the paper? Explain the Transformer architecture. \
# Can this be applied to text & images? Also, provide references for your answer to each question."
# query = "What was my last question?"
# st = time()
# result = neo4j.query(q=query,
#                     topK=10)
# print("Time taken to query : ", time()-st)
# # for record in result.records:
# #     print(record.data())

llm = ChatOpenAI(model_name="gpt-4o-mini", 
                 temperature=0,
                 openai_api_key=os.getenv('openai'))

# ---------------------------------------------------------- #

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
#                 "\n".join([record.data()['c.text'][:1000] for record in result.records][:4])  # Limit each chunk to 1000 chars and take top 4 chunks
#                 + "\n\nAnswer the question based on the above context." 
#                 + "\nMetadata for each context chunk is as follows:\n" +
#                 "\n".join([f"Chunk from page {record.data()['c.page']}" for record in result.records][:4]))
# ]
# response = llm.invoke(messages)
# print(response.content)


# ---------------------------------------------------------- #
prompt = ChatPromptTemplate([
    ("system", "You are a network graph maker who extracts terms and their relations from a given context.\n"
        "You are provided list of Document objects which are splits of a larger document. Your task is to extract the ontology "
        "of terms mentioned in the given context. These terms should represent the key concepts as per the context.\n\n"
        "Thought 1: While traversing through each sentence, think about the key terms mentioned in it.\n"
        "\tTerms may include object, entity, location, organization, person, condition, acronym, documents, service, concept, etc.\n"
        "\tTerms should be as atomistic as possible.\n\n"
        "Thought 2: Think about how these terms can have one-on-one relations with other terms.\n"
        "\tTerms that are mentioned in the same sentence or the same paragraph are typically related to each other.\n"
        "\tTerms can be related to many other terms."
        "Thought 3: Find out the relation between each related pair of terms.\n\n"
        "Format your output as a list of JSON objects. Each element of the list contains a pair of terms "
        "and the relation between them. Example output:\n"
        "[\n"
            "  {{\n"
            "    \"node_1\": \"A concept from extracted ontology\",\n"
            "    \"node_2\": \"A related concept from extracted ontology\",\n"
            "    \"edge\": \"relationship between the two concepts, node_1 and node_2 in one or two sentences\"\n"
            "  }}, ...\n" 
            "]\n\n"),
])

graphTransformer = LLMGraphTransformer(llm=llm, prompt=prompt)
# Sending just a single document chunk
graph_documents = graphTransformer.convert_to_graph_documents([doc_splits[0]])
for docs in graph_documents:
    pprint(docs.nodes)
    print("-----")
    pprint(docs.relationships)
    print("-----")
    pprint(docs.source)
    break

# Cleanup
# neo4j.delete_all_relations()
# neo4j.delete_all_nodes()