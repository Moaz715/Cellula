from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


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
        input = self.processor(img, return_tensors="pt")
        out = self.model.generate(**input)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption