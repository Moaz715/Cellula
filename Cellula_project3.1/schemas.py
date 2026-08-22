# schemas.py
from pydantic import BaseModel
from typing import List, Optional

class UserInput(BaseModel):
    query: str
    
class Response(BaseModel):
    answer: str
    
class IngestRequest(BaseModel):
    urls: Optional[List[str]] = []
    wiki_topics: Optional[List[str]] = []