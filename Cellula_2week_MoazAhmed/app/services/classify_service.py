from app.core.interfaces import TextClassifierInterface

class DistilBERTClassifier(TextClassifierInterface):
    def __init__(self):
        # We call load_model() when the class is created so it is ready for web traffic
        self.load_model()
        
    def load_model(self) -> None:
        """Blueprint method to handle downloading or loading weights into memory."""
        # PLACEHOLDER WHAT TO ADD LATER: 
        # 1. Import AutoTokenizer and AutoModelForSequenceClassification from 'transformers'.
        # 2. Load your fine-tuned LoRA DistilBERT weights from your local disk.
        # Example: self.tokenizer = AutoTokenizer.from_pretrained("./my-lora-model")
        # Example: self.model = AutoModelForSequenceClassification.from_pretrained("./my-lora-model")
        pass

    def predict(self, text: str) -> str:
        """Blueprint method that MUST accept a string and MUST return a string label."""
        # PLACEHOLDER WHAT TO ADD LATER:
        # 1. Pass the 'text' string through self.tokenizer()
        # 2. Feed the tokenized input into self.model()
        # 3. Apply torch.argmax() to the output logits to get the predicted class ID.
        # 4. Map the ID to your string (e.g., if ID == 1 return "Toxic", else "Safe")
        
        # Returning a dummy value so you can test the CSV database right now
        return "toxic_placeholder"
    
    