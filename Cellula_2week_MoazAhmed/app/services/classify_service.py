import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from peft import PeftModel
from app.core.interfaces import TextClassifierInterface

class DistilBERTClassifier(TextClassifierInterface):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.labels = [
            "safe", "violent crimes", "non-violent crimes", "unsafe",
            "suicide & self-harm", "elections", "sex-related crimes", 
            "child sexual exploitation", "unknown s-type"
        ]
        
        self.tokenizer = None
        self.model = None
        
        self.load_model()
        
    def load_model(self) -> None:
        """Loads the base model and attaches the custom LoRA weights."""
        print("Loading DistilBERT Tokenizer...")
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        
        print("Loading DistilBERT Base Model...")
        base_model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=9
        )
        
        print("Attaching LoRA Adapters...")
        lora_path = "./distilbert_lora_toxic_model2"
        self.model = PeftModel.from_pretrained(base_model, lora_path)
        
        self.model.to(self.device)
        self.model.eval()
        print("DistilBERT Classifier is ready!")

    def predict(self, text: str) -> str:
        """Accepts a string, runs inference, and returns the string label."""
        with torch.inference_mode():
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=128
            ).to(self.device)
            
            outputs = self.model(**inputs)
            
            predicted_class_id = torch.argmax(outputs.logits, dim=1).item()
            
            return self.labels[predicted_class_id]