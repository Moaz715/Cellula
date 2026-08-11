# schemas.py
from pydantic import BaseModel
from typing import List, Optional

class ChatMessage(BaseModel):
    role: str       
    content: str    
    intent: Optional[str] = None  

class UserRequest(BaseModel):
    prompt: str                 
    history: List[ChatMessage] 

class CodeExecutionRequest(BaseModel):
    code: str                   

class CodeExecutionResponse(BaseModel):
    output: str                 
    success: bool               

class SolutionRequest(BaseModel):
    query: str                  
    code: str