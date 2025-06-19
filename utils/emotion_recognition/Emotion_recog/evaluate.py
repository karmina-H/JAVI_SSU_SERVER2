# evaluate.py
import torch
from transformers import Wav2Vec2Processor
from model import Wav2Vec2ForEmotion
from dataset import EmotionDataset
from torch.utils.data import DataLoader


def collate_fn(batch):
    input_values = [item["input_values"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = torch.tensor([item["labels"] for item in batch])
    # pad sequences
    input_values = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True)
    return {
        "input_values": input_values,
        "attention_mask": attention_mask,
        "labels": labels
    }

# 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label2id = {
    "Happiness": 0,
    "Surprise": 1,
    "Neutral": 2,
    "Fear": 3,
    "Disgust": 4,
    "Anger": 5,
    "Sadness": 6
}
id2label = {v: k for k, v in label2id.items()}

# 모델 및 Processor 불러오기
model = Wav2Vec2ForEmotion(num_labels=len(label2id))
model.load_state_dict(torch.load("emotion_model.pt", map_location=device))
model.to(device)
model.eval()

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

# 테스트셋 로딩
test_dataset = EmotionDataset("evaluate.csv", processor, label2id)
test_loader = DataLoader(test_dataset, batch_size=8, collate_fn=collate_fn)

# 정확도 계산
correct = 0
total = 0

with torch.no_grad():
    for batch in test_loader:
        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_values=input_values, attention_mask=attention_mask)
        preds = torch.argmax(outputs["logits"], dim=-1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"테스트 정확도: {accuracy * 100:.2f}%")
