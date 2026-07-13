from pydantic import BaseModel
    
class CaptionResponse(BaseModel):
    img_caption: str
    
