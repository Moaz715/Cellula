import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from peft import PeftModel
from app.core.interfaces import TextClassifierInterface

def init_lstm_weights(lstm):
    for name, param in lstm.named_parameters():
        if "weight_ih" in name:
            nn.init.xavier_uniform_(param.data)
        elif "weight_hh" in name:
            nn.init.orthogonal_(param.data)
        elif "bias" in name:
            param.data.fill_(0)
            hidden_size = param.shape[0] // 4
            param.data[hidden_size:2*hidden_size].fill_(1.0)

class ToxicLSTM(nn.Module):
    def __init__(self, embedding_weights, hidden_dim=256, num_classes=9, dropout=0.4):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embedding_weights, freeze=True, padding_idx=0
        )
        self.lstm = nn.LSTM(
            input_size=embedding_weights.shape[1],
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        init_lstm_weights(self.lstm)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 4, num_classes)

    def forward(self, input_ids, attention_mask):
        lengths = attention_mask.sum(dim=1).cpu()
        embedded = self.dropout(self.embedding(input_ids))
        
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        seq_len = lstm_out.size(1)
        mask = attention_mask[:, :seq_len].unsqueeze(-1).float()

        max_pool = lstm_out.masked_fill(mask == 0, -1e9).max(dim=1)[0]
        avg_pool = (lstm_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        pooled = torch.cat([max_pool, avg_pool], dim=1)
        return self.classifier(self.dropout(pooled))

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
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", torch_dtype=torch.float16, low_cpu_mem_usage=True)
        
        print("Loading DistilBERT Base Model...")
        base_model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=9
        )
        
        print("Attaching LoRA Adapters...")
        lora_path = "./my_lora_adapter"
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