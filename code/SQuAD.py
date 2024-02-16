import torch
from transformers import BertTokenizer, BertForQuestionAnswering
from torch.utils.data import DataLoader, TensorDataset
import json
import numpy as np
import matplotlib.pyplot as plt

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

def load_and_preprocess_data(file_path):
    with open(file_path) as f:
        squad_data = json.load(f)

    questions, contexts = [], []
    for article in squad_data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                if qa['answers']:
                    questions.append(question)
                    contexts.append(context)
    return questions, contexts

def tokenize_and_create_loader(questions, contexts, batch_size=8):
    inputs = tokenizer(questions, contexts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'])
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader

def evaluate_model(model, dataloader):
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    for batch in dataloader:
        input_ids, attention_mask, token_type_ids = batch
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        total_predictions += input_ids.size(0)
    accuracy = correct_predictions / total_predictions
    return accuracy

def main():
    questions, contexts = load_and_preprocess_data('dev-v2.0.json')
    dataloader = tokenize_and_create_loader(questions, contexts)
    accuracy = evaluate_model(model, dataloader)
    print(f"Model Accuracy: {accuracy}")

    plt.figure(figsize=(5, 5))
    plt.bar(['Accuracy'], [accuracy], color=['blue'])
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Model Performance')
    plt.ylim(0, 1)
    plt.show()

if __name__ == "__main__":
    main()
