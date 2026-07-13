from fastapi import UploadFile, File, APIRouter, HTTPException
from app.schemas.caption import CaptionResponse
from app.services.caption_service import BLIP1CaptionService
import io
from PIL import Image

router = APIRouter()

blip_service = BLIP1CaptionService()

@router.post("/caption", response_model=CaptionResponse)
def generate_caption(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/gif"]:
        raise HTTPException(status_code=400, detail="File type not supported")
    
    
    contents = file.file.read()
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    caption = blip_service.generate_caption(pil_image)
    
    
    return {"img_caption": caption}
    
    