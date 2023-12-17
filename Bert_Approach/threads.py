import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from joblib import load
import torch
from concurrent.futures import ProcessPoolExecutor
import time

# record the start time
start_time = time.time()

# set process numbers
processes = 4

df = pd.read_csv('bert_test_gpt.csv')
texts = df['text'].tolist()

# assign text to each process
texts_per_process = 100
texts_for_each_process = [texts[i:i + texts_per_process] for i in range(0, len(texts), texts_per_process)]

texts_for_each_process = texts_for_each_process[:processes]

def predict(texts):
    model_start_time = time.time()

    # set device
    device = torch.device('cpu')

    model = BertForSequenceClassification.from_pretrained('./results/trained_model')
    tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')

    model.to(device)

    # load LabelEncoder
    labelencoder = load('labelencoder.joblib')

    # print the time to load the model
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
            predicted_label = labelencoder.inverse_transform([prediction.item()])[0]
            predictions.append(predicted_label)

    return predictions

# use ProcessPoolExecutor to run them concurrently
with ProcessPoolExecutor(max_workers=processes) as executor:
    results = executor.map(predict, texts_for_each_process)

# print predict result
for i, process_results in enumerate(results):
    print(f"Process {i+1} results:")
    for text, predicted_label in zip(texts_for_each_process[i], process_results):
        print(f"Text: {text}\nPredicted label: {predicted_label}\n")

# compute and print time used
end_time = time.time()
total_time = end_time - start_time
print(f"Total execution time: {total_time} seconds")
