import os
from time import time
import json
from dotenv import load_dotenv
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from .chat_graph import Neo4j
from .chat_data_model import TurnMemoryOutput

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


def run_query(user_message: str, llm, structured_llm) -> str:
   relevant_nodes = neo4j.get_relevant_nodes(query=user_message)
   if relevant_nodes:
        formatted_nodes = "\n".join([
            f"- [{node['type']}] {node['content']} (relevance: {node['score']:.2f})"
            for node in relevant_nodes
        ])
        context_section = f"""
                           ## Retrieved Memory Context
                           The following facts were retrieved from our conversation history, 
                           ordered by relevance to the current query:

                           {formatted_nodes}

                           ## Instructions
                           1. PRIORITIZE the retrieved context above when it directly answers the question.
                           2. If the context is a callback ("what did you say...", "going back to..."), 
                              answer from the retrieved context.
                           3. If the context is relevant but incomplete, supplement with your knowledge 
                              and clearly distinguish: "From our conversation: X. Additionally: Y."
                           4. If the context is irrelevant to the query, ignore it and answer normally.
                           """
   else:
      context_section = """
                        ## Retrieved Memory Context
                        No relevant context found from previous conversation.
                        Answer the question using your general knowledge.
                        """
   prompt = [
      ("system", f"""
      You are a helpful assistant with graph-based conversational memory.
      You have access to semantically retrieved facts from previous conversation turns.
      
      {context_section}
      """),
      ("human", user_message)
   ]
   response = llm.invoke(prompt)
   llm_response = response.content

   current_turn = { 
                "user_message": user_message,
                "llm_response": llm_response
                }
   existing_nodes = neo4j.view_all_nodes()
   open_questions = neo4j.get_open_questions()

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
      - user_message    : the query entered by user
      - llm_response    : the LLM's response for the query
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
      user_message: {current_turn["user_message"]}
      llm_response: {current_turn["llm_response"]}
      existing_nodes: {existing_nodes}
      open_questions: {open_questions}
      """)
   ]
   memory_response = structured_llm.invoke(memory_prompt)

   neo4j.add_turn(memory_response)
   return llm_response


# run_query("Explain US Treasury in 100 words", llm, structured_llm)   
# run_query("Explain US Federal bank in 100 words", llm, structured_llm)
# run_query("Connection between US Treasury and US Fed in 100 words.", llm, structured_llm)