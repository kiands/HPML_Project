import pandas as pd
from transformers import BertTokenizer
import numpy as np
import onnxruntime as ort
from joblib import load
import time
from concurrent.futures import ProcessPoolExecutor

# set process numbers
processes = 4

df = pd.read_csv('bert_test_gpt.csv')
texts = df['text'].tolist()

texts_per_process = 100
texts_for_each_process = [texts[i:i + texts_per_process] for i in range(0, len(texts), texts_per_process)]

texts_for_each_process = texts_for_each_process[:processes]

def predict_onnx(texts):
    start_time = time.time()

    tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')

    model_load_start = time.time()

    onnx_model_path = './onnx/model.onnx'

    session = ort.InferenceSession(onnx_model_path)
    model_load_end = time.time()
    model_loading_time = model_load_end - model_load_start

    labelencoder = load('labelencoder.joblib')

    predictions = []
    total_attention_time = 0

    for text in texts:
        inputs = tokenizer("Text: " + text, return_tensors="pt", padding='max_length', truncation=True, max_length=128)
        input_ids = inputs["input_ids"].numpy()
        attention_mask = inputs["attention_mask"].numpy()

        attention_start = time.time()  # start to record attention_mask conversion time
        attention_mask_float = attention_mask.astype(np.float32)
        attention_end = time.time()  # stop attention_mask conversion timer
        total_attention_time += attention_end - attention_start  # accumulate time

        outputs = session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask_float})
        logits = outputs[0]
        pred = np.argmax(logits, axis=-1)
        predicted_label = labelencoder.inverse_transform(pred)[0]
        predictions.append(predicted_label)

    total_time = time.time() - start_time  # total run time
    effective_time = total_time - total_attention_time  # deduct attention_mask conversion time

    return predictions, model_loading_time, effective_time

# use ProcessPoolExecutor
with ProcessPoolExecutor(max_workers=processes) as executor:
    futures = [executor.submit(predict_onnx, texts) for texts in texts_for_each_process]
    for i, future in enumerate(futures):
        process_results, model_loading_time, effective_time = future.result()
        print(f"Process {i+1} results:")
        print(f"Model loading time: {model_loading_time:.2f} seconds")
        print(f"Total effective running time (excluding attention_mask conversion): {effective_time:.2f} seconds\n")
        for text, predicted_label in zip(texts_for_each_process[i], process_results):
            print(f"Text: {text}\nPredicted label: {predicted_label}\n")

