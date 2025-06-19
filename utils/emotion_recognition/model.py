# model.py
from transformers import Wav2Vec2Model
import torch.nn as nn
import torch

class Wav2Vec2ForEmotion(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.classifier = nn.Sequential(
            nn.Linear(self.wav2vec.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_values, attention_mask=None, labels=None):
        outputs = self.wav2vec(input_values=input_values, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {'loss': loss, 'logits': logits}
