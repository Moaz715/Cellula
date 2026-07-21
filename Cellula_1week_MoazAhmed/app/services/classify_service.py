import torch
import torch.nn as nn
from transformers import DistilBertTokenizer
from app.core.interfaces import TextClassifierInterface
import os

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

class ToxicLSTMClassifier(TextClassifierInterface):
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.id2label = {}
        self.load_model()

    def load_model(self) -> None:
        bundle_path = os.path.join(os.getcwd(), "toxic_lstm_deployment.pth")
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"Deployment bundle missing at: {bundle_path}")
            
        bundle = torch.load(bundle_path, map_location=torch.device('cpu'))
        self.id2label = bundle['id2label']
        
        self.model = ToxicLSTM(
            embedding_weights=bundle['embedding_weights'],
            hidden_dim=bundle['hidden_dim'],
            num_classes=bundle['num_classes']
        )
        
        self.model.load_state_dict(bundle['model_state_dict'])
        self.model.eval()
        
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        print("LSTM Classifier ready!")

    def predict(self, text: str) -> str:
        inputs = self.tokenizer(
            text, max_length=128, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        with torch.no_grad():
            logits = self.model(inputs['input_ids'], inputs['attention_mask'])
            prediction_idx = logits.argmax(-1).item()
            
        return self.id2label[prediction_idx]

