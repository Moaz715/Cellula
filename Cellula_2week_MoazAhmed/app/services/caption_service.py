from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch


class BLIP1CaptionService():
    
    def __init__(self):
        self.processor = None
        self.model = None
        self._load_model()
        
        
    def _load_model(self):
        model_name = "Salesforce/blip-image-captioning-base"
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        
        
    def generate_caption(self, img: Image.Image) -> str:
        with torch.inference_mode():
            inputs = self.processor(img, return_tensors="pt")
            
            out = self.model.generate(**inputs, max_new_tokens=25)
            
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            return caption