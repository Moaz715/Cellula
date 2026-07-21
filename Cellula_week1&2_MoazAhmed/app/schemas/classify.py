from pydantic import BaseModel

class ClassifyRequest(BaseModel):
    text: str
    
    
class ClassifyResponse(BaseModel):
    text: str
    prediction: str
