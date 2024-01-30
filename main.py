import argparse
import numpy as np
import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import MultiheadAttention
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from tqdm import tqdm, trange
from transformers import AutoModel, ResNetModel, AutoFeatureExtractor, AutoTokenizer


bert_model_path = './pretrained_models/bert-base-uncased'
resnet_model_path = './pretrained_models/resnet-50'


# 数据处理部分

def get_data_item(guid, tag, emotion_to_index=None, is_test=False):
    image_path = f'./data/{int(guid)}.jpg'
    text_path = f'./data/{int(guid)}.txt'
    
    image = Image.open(image_path)
    
    with open(text_path, 'r', encoding='gb18030', errors='replace') as f:
        text = f.readline().strip()
    
    emotion_to_index = {'negative': 0, 'neutral': 1, 'positive': 2}
    
    if is_test:
        tag = None
    else:
        tag = emotion_to_index[tag]
    
    return {'guid': int(guid), 'tag': tag, 'image': image, 'text': text}

class DatasetProcessor:
    def __init__(self, train_file, test_file, batch_size, resnet_model_path, bert_model_path):
        self.train_file = train_file
        self.test_file = test_file
        self.batch_size = batch_size
        self.resnet_model_path = resnet_model_path
        self.bert_model_path = bert_model_path
        self.label_to_id = {'negative': 0, 'neutral': 1, 'positive': 2}

    def load_data(self, file_path, is_test=False):
        data = pd.read_csv(file_path)
        datasets = [get_data_item(row['guid'], row['tag'], self.label_to_id, is_test) for _, row in data.iterrows()]
        return datasets

    def get_dataloaders(self):
        train_datasets = self.load_data(self.train_file)
        test_datasets = self.load_data(self.test_file, is_test=True)

        train_datasets, valid_datasets = train_test_split(train_datasets, test_size=0.1, random_state=42)

        train_dataloader = DataLoader(dataset=train_datasets, collate_fn=self.collate_fn, batch_size=self.batch_size, shuffle=True, num_workers=4)
        valid_dataloader = DataLoader(dataset=valid_datasets, collate_fn=self.collate_fn, batch_size=self.batch_size, shuffle=False, num_workers=4)
        test_dataloader = DataLoader(dataset=test_datasets, collate_fn=self.collate_fn, batch_size=self.batch_size, shuffle=False, num_workers=4) # type: ignore

        return train_dataloader, valid_dataloader, test_dataloader

    def collate_fn(self, datasets):
        feature_extractor = AutoFeatureExtractor.from_pretrained(self.resnet_model_path)
        tokenizer = AutoTokenizer.from_pretrained(self.bert_model_path)

        guid = [data['guid'] for data in datasets]
        tag = [data['tag'] for data in datasets]
        tag = None if tag[0] is None else torch.LongTensor(tag)
        images = [data['image'] for data in datasets]
        images = feature_extractor(images, return_tensors="pt")
        texts = [data['text'] for data in datasets]
        texts = tokenizer(texts, return_tensors="pt", padding='max_length', truncation=True, max_length=64)

        return guid, tag, images, texts

def read_data(batch_size, resnet_model_path, bert_model_path):
    processor = DatasetProcessor('train.txt', 'test_without_label.txt', batch_size, resnet_model_path, bert_model_path)
    return processor.get_dataloaders()


# 模型部分

class MultimodalClassifier(nn.Module):
    def __init__(self, num_labels=3):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(bert_model_path)
        self.visual_encoder = ResNetModel.from_pretrained(resnet_model_path)
        self.flatten = nn.Flatten()
        self.text_linear = nn.Linear(self.text_encoder.config.hidden_size, 768)  
        self.image_linear = nn.Linear(2048, 768)  
        self.attention = nn.MultiheadAttention(embed_dim=768 * 2, num_heads=8)
        self.text_classifier = nn.Linear(768, num_labels)  
        self.image_classifier = nn.Linear(768, num_labels)  
        self.combined_classifier = nn.Linear(768 * 2, num_labels)  

    def encode_text(self, text):
        text_output = self.text_encoder(**text)
        return text_output.last_hidden_state[:, 0, :]

    def encode_image(self, image):
        img_output = self.visual_encoder(**image)  # type: ignore
        img_feature = img_output.last_hidden_state.view(-1, 49, 2048).max(1)[0]
        return img_feature

    def forward(self, text=None, image=None):
        if text is not None and image is not None:
            text_feature = self.encode_text(text)
            img_feature = self.encode_image(image)
            text_feature = self.text_linear(text_feature)
            img_feature = self.image_linear(img_feature)
            combined_features = torch.cat((text_feature, img_feature), dim=1)

            attention_output, _ = self.attention(combined_features.unsqueeze(0), combined_features.unsqueeze(0), combined_features.unsqueeze(0))
            logits = self.combined_classifier(attention_output.squeeze(0))
        elif text is not None:
            text_feature = self.encode_text(text)
            text_feature = self.text_linear(text_feature)
            logits = self.text_classifier(text_feature)
        else:  # image is not None
            img_feature = self.encode_image(image)
            img_feature = self.image_linear(img_feature)
            logits = self.image_classifier(img_feature)

        return logits


# 训练及预测部分
    
def train_step(model, batch, criterion, optimizer, device, text_only, image_only):
    model.train()
    a, b_labels, b_imgs, b_text = batch
    b_labels, b_imgs, b_text = b_labels.to(device), b_imgs.to(device), b_text.to(device)
    model.zero_grad()
    
    if text_only:
        b_logits = model(text=b_text, image=None)
    elif image_only:
        b_logits = model(text=None, image=b_imgs)
    else:
        b_logits = model(text=b_text, image=b_imgs)
    
    loss = criterion(b_logits, b_labels)
    loss.backward()
    optimizer.step()
    # model.zero_grad()
    
    return loss.item(), torch.max(b_logits, 1)[1], b_labels

def eval_step(model, batch, criterion, device, text_only, image_only):
    model.eval()
    with torch.no_grad():
        a, b_labels, b_imgs, b_text = batch
        b_labels, b_imgs, b_text = b_labels.to(device), b_imgs.to(device), b_text.to(device)
        
        if text_only:
            b_logits = model(text=b_text, image=None)
        elif image_only:
            b_logits = model(text=None, image=b_imgs)
        else:
            b_logits = model(text=b_text, image=b_imgs)
        
        loss = criterion(b_logits, b_labels)
        
        return loss.item(), torch.max(b_logits, 1)[1], b_labels

def compute_metrics(preds, labels):
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    return acc, f1, precision, recall

def train(train_dataloader, valid_dataloader, model, epochs, weight_decay, learning_rate, text_only, image_only):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_loss, train_preds, train_labels = [], [], []
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            loss, preds, labels = train_step(model, batch, criterion, optimizer, device, text_only, image_only)
            train_loss.append(loss)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_acc, train_f1, train_precision, train_recall = compute_metrics(train_preds, train_labels)
        print(f"Epoch {epoch+1} Train Loss: {np.mean(train_loss):.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f}")

        val_loss, val_preds, val_labels = [], [], []
        for batch in tqdm(valid_dataloader, desc=f"Validating Epoch {epoch+1}/{epochs}"):
            loss, preds, labels = eval_step(model, batch, criterion, device, text_only, image_only)
            val_loss.append(loss)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

        val_acc, val_f1, val_precision, val_recall = compute_metrics(val_preds, val_labels)
        print(f"Epoch {epoch+1} Val Loss: {np.mean(val_loss):.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

    print('\n*** Training completed ***\n')


def predict_and_replace(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index_to_emotion = {0: 'negative', 1: 'neutral', 2: 'positive'}
    model = model.to(device).eval()

    replacements = []
    for batch in tqdm(dataloader):
        _, b_labels, b_imgs, b_text = batch
        b_text, b_imgs = b_text.to(device), b_imgs.to(device)

        with torch.no_grad():
            b_logits = model(text=b_text, image=b_imgs).detach().cpu()

        batch_predictions = torch.argmax(b_logits, dim=-1).tolist()
        replacements.extend([index_to_emotion[idx] for idx in batch_predictions])

    file_path = 'test_without_label.txt'
    new_file_path = 'test_with_label.txt'

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines[1:]):
        if 'null' in line and i < len(replacements):
            lines[i + 1] = line.replace('null', replacements[i])

    with open(new_file_path, 'w') as new_file:
        new_file.writelines(lines)


# 参数处理与设置
        
def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal sentiment analysis task")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for the optimizer")
    parser.add_argument('--text_only', action='store_true', help='use text only')
    parser.add_argument('--image_only', action='store_true', help='use image only')
    return parser.parse_args()

def set_random_seeds(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# 主函数
    
def main():
    args = parse_args()
    set_random_seeds()
    train_dataloader, valid_dataloader, test_dataloader = read_data(args.batch_size, resnet_model_path, bert_model_path)

    model = MultimodalClassifier()

    train(train_dataloader, valid_dataloader, model, args.epochs, args.weight_decay, args.learning_rate, args.text_only, args.image_only)

    model_path = f'./my_model/my_model.pth'
    torch.save(model, model_path)

    predict_and_replace(model, test_dataloader)

# # 加载训练好的pth模型直接进行预测
#     train_dataloader, valid_dataloader, test_dataloader = read_data(batch_size, resnet_model_path, bert_model_path)
#     model_path = f'./my_model/both_model_15epochs.pth'
#     model = torch.load(model_path)
#     predict_and_replace(model, test_dataloader)


# # 用于自动化训练三个模式的训练
#     configurations = [
#         {'text_only': True, 'image_only': False, 'model_name': 'text_only'},
#         {'text_only': False, 'image_only': True, 'model_name': 'image_only'},
#         {'text_only': False, 'image_only': False, 'model_name': 'text&image'}
#     ]

#     for config in configurations:
#         parser = argparse.ArgumentParser(description="Multimodal sentiment analysis task")
#         parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
#         parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for the optimizer")
#         parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for the optimizer")
#         parser.add_argument('--text_only', action='store_true', help='only use text')
#         parser.add_argument('--image_only', action='store_true', help='only use image')
#         args = parser.parse_args([])  

#         args.text_only = config['text_only']
#         args.image_only = config['image_only']

#         set_random_seeds()
#         train_dataloader, valid_dataloader, test_dataloader = read_data(batch_size, resnet_model_path, bert_model_path)
#         model = MultimodalClassifier()

#         train(train_dataloader, valid_dataloader, model, args.epochs, args.weight_decay, args.learning_rate, args.text_only, args.image_only)

#         model_path = f'./my_model/{config["model_name"]}_model.pth'
#         torch.save(model, model_path)

#         predict_and_replace(model, test_dataloader)


if __name__ == "__main__":
    main()
