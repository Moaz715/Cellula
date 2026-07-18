from fastapi import UploadFile, File, APIRouter, HTTPException
from app.schemas.classify import ClassifyRequest, ClassifyResponse
from app.services.classify_service import DistilBERTClassifier
from app.services.caption_service import BLIP1CaptionService
from app.db.session import SQLiteDatabaseManager, CSVDatabaseManager
from PIL import Image
import io


router = APIRouter()
classifier_service = DistilBERTClassifier()
blip_service = BLIP1CaptionService()
db = CSVDatabaseManager()

@router.post("/classify", response_model=ClassifyResponse)
def classify_text(request: ClassifyRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty!")
        
    prediction = classifier_service.predict(request.text)
    
    db.log_entry(input_type="text", content=request.text, prediction=prediction)
    
    return{
        "text": request.text,
        "prediction": prediction
    }
    

@router.post("/classifyImage", response_model=ClassifyResponse)
def generate_caption(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/gif"]:
        raise HTTPException(status_code=400, detail="File type not supported")
    
    
    contents = file.file.read()
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    caption = blip_service.generate_caption(pil_image)
    
    
    
    if not caption.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty!")
        
    prediction = classifier_service.predict(caption)
    
    db.log_entry(input_type="image", content=caption, prediction=prediction)
    
    return{
        "text": caption,
        "prediction": prediction
    }
