import os
from dotenv import load_dotenv
from pathlib import Path
from dataclasses import dataclass
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from chat_system.chat_graph import Neo4j
from chat_system.chat_data_model import TurnMemoryOutput
from chat_system.chat_main import run_query

load_dotenv(Path(__file__).parent.parent / ".env")

@dataclass
class EvalTurn:
    user: str
    expected_keywords: List[str]
    is_filler: bool = False
    is_callback: bool = False  

EVAL_CONVERSATION = [
    EvalTurn("What is the US Treasury?",
             ["federal", "finances", "government"],
             is_filler=False),

    EvalTurn("What does the IRS do within the Treasury?",
             ["taxes", "IRS", "revenue"],
             is_filler=False),

    # ── Filler: unrelated topics to dilute flat history ──────────
    EvalTurn("Tell me about the Roman Empire.",
             [], is_filler=True),

    EvalTurn("Who was Julius Caesar?",
             [], is_filler=True),

    EvalTurn("Explain photosynthesis.",
             [], is_filler=True),

    EvalTurn("What is the capital of Australia?",
             [], is_filler=True),

    EvalTurn("Tell me about black holes. Who is right Einstein or hawking?",
             [], is_filler=True),

    EvalTurn("How does a combustion engine work?",
             [], is_filler=True),

    # ── Callbacks ──────────────────
    EvalTurn("What was the capital of Australia you said, is it Paris? Which year was it decided?",
             ["1913", "Canberra", "Australia"],
             is_callback=True),

    EvalTurn("Why did Julius kill Brutus?",
             ["Conspiracy", "Roman Republic", "Caesar"],
             is_callback=True),

    EvalTurn("Summarize only what we discussed about the US Treasury earlier.",
             ["Treasury", "IRS", "taxes", "finances", "bonds"],
             is_callback=True),
]


class FlatHistoryChat:
    def __init__(self, llm):
        self.llm = llm
        self.history = []

    def chat(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})

        messages = [("system", "You are a helpful assistant.")] + [
            (h["role"], h["content"]) for h in self.history
        ]

        response = self.llm.invoke(messages)
        self.history.append({"role": "assistant", "content": response.content})
        return response.content


# ── Scoring –––––––––––––––––––––––––––––––––––––––––--––––––––
def score_response(response: str, keywords: List[str]) -> dict:
    if len(keywords) == 0:
        return { "score": 0.0 , "misses": 0.0, "hits": 0.0}
    response_lower = response.lower()
    hits   = [kw for kw in keywords if kw.lower() in response_lower]
    misses = [kw for kw in keywords if kw.lower() not in response_lower]
    return {
        "score": len(hits) / len(keywords),
        "hits": hits,
        "misses": misses
    }


# ── Logging –––––––––––––––––––––––––––––––––––––––––--––––––––
def write_responses_file(file_path: str, q_id: int, q: str, response: str):
    clean_q = f"Q{q_id}. " + q.rstrip('\r\n') + '\n'
    clean_response = "Response: " + response.rstrip('\r\n') + '\n'
    
    with open(file=file_path, mode="a", encoding="utf-8") as file:
        file.write(clean_q)
        file.write(clean_response)


# ── Run Eval –––––––––––––––––––––––––––––––––––––––––––––––––––
def run_eval():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    structured_llm = llm.with_structured_output(TurnMemoryOutput)
    neo4j = Neo4j("neo4j://0.0.0.0:7687", 
                  "neo4j", 
                  os.getenv("NEO4J_Password"))

    # Clean slate for graph system
    neo4j.delete_all_nodes()

    flat = FlatHistoryChat(llm)
    flat_scores  = []
    graph_scores = []

    print("=" * 60)
    print("Flat History vs Graph-based history")
    print("=" * 60)

    for i, turn in enumerate(EVAL_CONVERSATION):
        print(f"\n--- Turn {i+1} ---")
        print(f"User: {turn.user}")

        # Flat system
        flat_response = flat.chat(turn.user)
        write_responses_file("flat-chat-responses.txt", i+1, turn.user, flat_response)
        print(f"Flat chat Response for turn {i+1} written to file.")
        flat_score = score_response(flat_response, turn.expected_keywords)
        flat_scores.append(flat_score["score"])

        # Graph system — reuse run_query from chat_main
        graph_response = run_query(turn.user, llm, structured_llm)
        write_responses_file("graph-chat-responses.txt", i+1, turn.user, graph_response)
        print(f"Graph chat Response for turn {i+1} written to file.")
        graph_score = score_response(graph_response, turn.expected_keywords)
        graph_scores.append(graph_score["score"])

        # Print results
        print(f"\nFlat  | score: {flat_score['score']:.2f} | misses: {flat_score['misses']}")
        print(f"Graph | score: {graph_score['score']:.2f} | misses: {graph_score['misses']}")
        # print(f"\nFlat  response: {flat_response}")
        # print(f"Graph response: {graph_response}")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print(f"Flat  avg score: {sum(flat_scores)/len(flat_scores):.2f}")
    print(f"Graph avg score: {sum(graph_scores)/len(graph_scores):.2f}")
    print("\nPer-turn breakdown:")
    for i, (f, g) in enumerate(zip(flat_scores, graph_scores)):
        winner = "GRAPH" if g > f else ("FLAT" if f > g else "TIE")
        print(f"  Turn {i+1}: Flat={f:.2f}  Graph={g:.2f}  -> {winner}")


if __name__ == "__main__":
    run_eval()