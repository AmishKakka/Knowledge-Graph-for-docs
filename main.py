import os
from typing import List
import arxiv
import time
from functools import partial
from dotenv import load_dotenv
from pathlib import Path
from document.doc_loader import DocumentLoader
from graph import Neo4j
from langchain_google_genai import ChatGoogleGenerativeAI
from chat_system.chat_data_model import TurnMemoryOutput
from chat_system.chat_main import run_query
from urllib.request import urlretrieve
print("Imported required files and packages...")


def get_research_papers(topics: List[str]):
    client = arxiv.Client()

    for query in topics:
        search = arxiv.Search(
            query=query,
            max_results=1,
            sort_by=arxiv.SortCriterion.Relevance
        )

        os.makedirs("papers", exist_ok=True)
        for paper in client.results(search):
            pdf_url = paper.pdf_url
            if not pdf_url:
                print(f"Skipping {paper.title}: no PDF URL available")
                continue
            print(f"Downloading: {paper.title}")
            urlretrieve(url=pdf_url, filename=f"papers/{paper.title}.pdf")
            time.sleep(1)
        print(f"All papers for {query} downloaded...")
        time.sleep(1)


if __name__ == "__main__":
    # Loading the environment variables
    load_dotenv(Path(__file__).parent / ".env")
    neo4j_pwd       = os.getenv("NEO4J_Password")
    documents_path  = "papers"
    neo4j           = Neo4j(
                        "neo4j://0.0.0.0:7687",    # Connecting to the Neo4j instance running inside your docker container
                        "neo4j", neo4j_pwd
                        )
    llm             = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                              temperature=0,
                              google_api_key=os.getenv('GOOGLE_API_KEY'))
    structured_llm  = llm.with_structured_output(TurnMemoryOutput)

    # First, download research papers from Arxiv
    # get_research_papers(topics=[
    #     "attention mechanism LLM",
    #     "knowledge graph QA",
    #     "Reinforement learning",
    #     "neural networks"
    # ])
    files = [f for f in os.listdir(documents_path) if os.path.isfile(os.path.join(documents_path, f))]

    # Chunk each research paper, embed the content, and then update the knowledge graph
    doc_loader = DocumentLoader()
    doc_splits = doc_loader.lazy_load_and_split([os.path.join(documents_path, f) for f in files])
    chunks, list_texts = doc_loader.create_chunks(doc_splits)

    # Adding nodes to the graph
    neo4j.add_document_nodes(chunks, list_texts)

    # Add relation between nodes
    neo4j.precedence_relationship()

    # Run queries 
    query       = partial(run_query, llm, structured_llm, neo4j)
    response    = query(user_message="Can you list all the 4 papers and give me a summary for all? Also, how are they related, if they are?")

    print("Query response:  ", response)