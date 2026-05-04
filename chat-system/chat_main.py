import os
from time import time
import json
from dotenv import load_dotenv
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from chat_graph import Neo4j
from chat_data_model import TurnMemoryOutput

load_dotenv(Path(__file__).parent.parent / ".env")
neo4j_pwd = os.getenv("NEO4J_Password")
# Connecting to Neo4j
neo4j = Neo4j(
            "neo4j://0.0.0.0:7687",    # Connecting to the Neo4j instance running inside your docker container
            "neo4j", neo4j_pwd
            )

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                              temperature=0,
                              google_api_key=os.getenv('GOOGLE_API_KEY'))
structured_llm = llm.with_structured_output(TurnMemoryOutput)

current_turn = {"turn_id": 1, 
                "user_message": "Explain to me the Transfomer architecture.",
                "llm_response": ""}
existing_nodes = []
open_questions = []
memory_prompt = [
    ("system", """
    You are a Conversational Memory Graph engine. Your job is to extract a 
    structured graph from a single conversation turn (one user message + one 
    LLM response), given the existing graph state as context.

    ## Node Types
    - Turn       : Represents one exchange (user + LLM). Has turn_id, timestamp.
    - Topic      : A subject, concept, or entity being discussed.
                   (e.g., "Transformer Architecture", "Paris", "Project Deadline")
    - Claim      : A specific fact or assertion made by either party.
                   (e.g., "Transformers process tokens in parallel")
    - Question   : An explicit or implicit question raised by the user.
                   (e.g., "Which architecture handles long sequences better?")
    - UserIntent : The underlying goal inferred from the conversation so far.
                   (e.g., "Compare ML architectures", "Plan a trip to France")

    ## Relationship Types
    - ASKS_ABOUT    : (Turn)     → (Topic)     [user introduced this topic]
    - ANSWERS_WITH  : (Turn)     → (Claim)     [LLM made this claim in response]
    - RAISES        : (Turn)     → (Question)  [this question was raised this turn]
    - RESOLVES      : (Turn)     → (Question)  [this turn answered a prior question]
    - REFINES       : (Question) → (Question)  [follow-up or narrowing of prior question]
    - CONTRASTS     : (Topic)    → (Topic)     [explicitly compared in conversation]
    - PART_OF       : (Claim)    → (Topic)     [this claim belongs to this topic]
    - FOLLOWS_FROM  : (Turn)     → (Turn)      [sequential linkage]
    - RELATED_TO    : (Topic)    → (Topic)     [topically connected, not contrasted]
    - UNRESOLVED    : (Question) → (Turn)      [question raised but not yet answered]

    ## Extraction Rules

    1. One Turn node per exchange. Always create it.

    2. Normalize Topic IDs to canonical full names.
       ✗ "it", "that model", "the city"
       ✓ "Transformer Architecture", "Paris", "GPT-4"

    3. Claims should be atomic — one idea per Claim node.
       ✗ "Transformers use attention and are faster than RNNs"  (two claims)
       ✓ "Transformers use self-attention mechanisms"
       ✓ "Transformers are faster than RNNs for long sequences"

    4. Only mark a Question as UNRESOLVED if the LLM response did not 
       address it in this turn.

    5. Reuse existing node IDs from the graph context if the same 
       entity/topic appears again. Do NOT create duplicate nodes.
       This is critical for graph coherence across turns.

    6. Infer UserIntent only if it has changed or become clearer this turn.
       Don't re-extract the same intent every turn.

    ## Input Format
    You will receive:
    - current_turn    : { turn_id, user_message, llm_response }
    - existing_nodes  : list of nodes already in the graph (id, type, summary)
    - open_questions  : list of Question nodes currently marked UNRESOLVED

    ## Output Format
    - new_nodes          : nodes to ADD to the graph
    - updated_nodes      : existing node IDs whose properties changed
    - new_relationships  : relationships to ADD
    - resolved_questions : Question node IDs now resolved (remove UNRESOLVED edge)
    - graph_summary      : 2-3 sentence natural language summary of the 
                           conversation state so far (for LLM context injection)
    """),
    ("human", f"""
    current_turn: {current_turn}
    existing_nodes: {existing_nodes}
    open_questions: {open_questions}
    """)
]

#  turn 1
st = time()
response1 = llm.invoke(current_turn["user_message"])
et1= time()
print("Time taken for LLM to respond to question: ", et1-st)
current_turn["llm_response"] = response1.content
graph_t1 = structured_llm.invoke(memory_prompt)
et2 = time()
print("Time taken to understand the current graph: ", et2-st)
print(graph_t1)

neo4j.add_turn(graph_t1)
existing_nodes = neo4j.view_all_nodes()
open_questions = neo4j.get_open_questions()
print(existing_nodes)
print(open_questions)


#  turn 2
current_turn["user_message"] = "How does this architecture compare to RNNs?"
current_turn["turn_id"] = 2
response2 = llm.invoke(current_turn["user_message"])
print(response2)
current_turn["llm_response"] = response2.content
graph_t2 = structured_llm.invoke(memory_prompt)

print(graph_t2)
neo4j.add_turn(graph_t2)
existing_nodes = neo4j.view_all_nodes()
open_questions = neo4j.get_open_questions()
print(existing_nodes)
print(open_questions)

