from pydantic import BaseModel, Field
from typing import List, Optional

class Node(BaseModel):
    id: str
    type: str
    properties: dict

class Relationship(BaseModel):
    source: str
    target: str
    type: str

class GraphOutput(BaseModel):
    nodes: List[Node]
    relationships: List[Relationship]