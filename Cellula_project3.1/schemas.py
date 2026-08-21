# schemas.py
from pydantic import BaseModel

class UserInput(BaseModel):
    query: str
    
class Response(BaseModel):
    answer: str