from pydantic import BaseModel
from typing import List, Optional
from enum import Enum

class NodeType(str, Enum):
    turn = "Turn"
    topic = "Topic"
    claim = "Claim"
    question = "Question"
    user_intent = "UserIntent"

class RelType(str, Enum):
    asks_about = "ASKS_ABOUT"
    answers_with = "ANSWERS_WITH"
    raises = "RAISES"
    resolves = "RESOLVES"
    refines = "REFINES"
    contrasts = "CONTRASTS"
    part_of = "PART_OF"
    follows_from = "FOLLOWS_FROM"
    related_to = "RELATED_TO"
    unresolved = "UNRESOLVED"

class MemoryNode(BaseModel):
    id: str
    type: NodeType
    content: str
    turn_id: int           
    embedding: Optional[List[float]] = None

class MemoryRelationship(BaseModel):
    source: str
    target: str
    type: RelType
    turn_id: int

class TurnMemoryOutput(BaseModel):
    new_nodes: List[MemoryNode]
    updated_node_ids: List[str]
    new_relationships: List[MemoryRelationship]
    resolved_question_ids: List[str]
    graph_summary: str