from pydantic import BaseModel

class ClassifyCreate(BaseModel):
    img_url : str
    
    
class ClassifyResponse(BaseModel):
    img_caption: str
    
