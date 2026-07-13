from fastapi import UploadFile, File, APIRouter, HTTPException
from app.schemas.classify import ClassifyCreate, ClassifyResponse


router = APIRouter()