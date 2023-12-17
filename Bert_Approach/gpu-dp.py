import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from joblib import load
import torch
import time

start_time = time.time()

# change the records amout here
texts_count = 400

df = pd.read_csv('bert_test_gpt.csv')
texts = df['text'].tolist()

texts = texts[:texts_count]

def predict(texts):
    # the timestamp we starts to load model
    model_start_time = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = BertForSequenceClassification.from_pretrained('./results/trained_model')
    tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')

    # use DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    model.to(device)

    # load the label encoder
    labelencoder = load('labelencoder.joblib')

    # print the time needed to load the model
    model_end_time = time.time()
    print(f"Model loaded in {model_end_time - model_start_time} seconds")

    predictions = []
    for new_text in texts:
        bert_input = f"Text: {new_text}"
        inputs = tokenizer(bert_input, padding=True, truncation=True, max_length=128, return_tensors="pt")
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1)
            predicted_label = labelencoder.inverse_transform([prediction.cpu().item()])[0]
            predictions.append(predicted_label)

    return predictions

# inference
predictions = predict(texts)

# print inference result
for text, predicted_label in zip(texts, predictions):
    print(f"Text: {text}\nPredicted label: {predicted_label}\n")

# compute the times
end_time = time.time()
total_time = end_time - start_time
print(f"Total execution time: {total_time} seconds")

