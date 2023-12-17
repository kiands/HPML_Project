import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from joblib import load
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
import time
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def predict(rank, world_size, texts):
    setup(rank, world_size)

    device = torch.device('cuda', rank)

    # load model and tokenizer
    model = BertForSequenceClassification.from_pretrained('./results/trained_model')
    tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
    labelencoder = load('labelencoder.joblib')

    model.to(device)
    model = DDP(model, device_ids=[rank])

    # data processing
    encoded_texts = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    dataset = TensorDataset(encoded_texts['input_ids'], encoded_texts['attention_mask'], encoded_texts['token_type_ids'])
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=16)

    # inference
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = {'input_ids': batch[0].to(device), 'attention_mask': batch[1].to(device), 'token_type_ids': batch[2].to(device)}
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1)
            predicted_label = labelencoder.inverse_transform(prediction.cpu().numpy())
            predictions.extend(predicted_label)

    # print result (optional)
    print(f"Rank {rank}, Predictions: {predictions}")

    cleanup()

def run_ddp_inference(texts):
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(predict, args=(world_size, texts), nprocs=world_size, join=True)

if __name__ == "__main__":
    start_time = time.time()

    # read text from csv
    df = pd.read_csv('bert_test_gpt.csv')
    texts = df['text'].tolist()

    # Change records amout here
    texts_count = 400
    texts = texts[:texts_count]

    # run DDP inference
    run_ddp_inference(texts)

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time} seconds")

