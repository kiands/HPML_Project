# Model: bert-base-uncased

**bert_test_gpu.csv**: data used to test the fine tuned model  
**bert_train_gpu.csv**: data used to fine tune the model  
**fine_tune_bert.ipynb**: notebook used to fine tune the bert model and convert it to onnx format  
**onnx_model_compatibility_test.ipynb**: used to test the onnx model  
**bert_base_uncased**: download the bert-base-uncased model and extract it under this folder  
**onnx**: folder to store onnx model  
**results**: used to store fine tuned model and save confusion matrix  
**labelencoder.joblib**: will be generated after each succeeded fine tune  
**threads.py**: CPU inference with multi process enabled, can output time  
**threads_onnx.py**: same function like threads.py, but switched pytorch to onnx runtime  
**gpu_single.py**: program for testing single GPU inference  
**gpu-gpu.py**: program for testing multi GPU data parallel inference  
**gpu-ddp.py**: program for testing multi GPU distributed data parallel inference

## Commands:
pip install -r requirements.txt
python threads.py
python threads_onnx.py
python gpu_single.py
python gpu-gpu.py
python gpu-ddp.py

## The following scripts can only be executed on multi GPU computers like NYU HPC  
ssh burst  
srun --account=ece_gy_9143-2023fa --cpus-per-task=8 --partition=n1c24m128-v100-4 --gres=gpu:v100:4 --time=04:00:00 --pty /bin/bash
python gpu-gpu.py
python gpu-ddp.py

## Inference speed

1. BERT CPU VS RTX 4080  
time to load model/time for CPU to inference/Memory bandwidth utilization

|          | 1 thread* 400records | 2 threads* 200records | 4 threads* 100records |
|----------|----------------------|-----------------------|-----------------------|
| Pytorch  | 0.61s/16.5s/26.71GB/s| 0.62s/4.89s/35.99GB/s | 0.67s/4.06s/50.32GB/s |
| ONNX     | 0.36s/9.25s/17.97GB/s| 0.45s/5.00s/11.48GB/s | 0.52s/4.17s/7.67GB/s  |

VS

|          | RTX 4080 on 400 records: 0.95s to load model, 1.37s to inference |
|----------|------------------------------------------------------------------|

2. BERT single V100/4*V100 DP/4*V100 DDP on 4000 records

|               | single | DP    | DDP   |
|---------------|--------|-------|-------|
| Time to load model | 2.97s  | 3.09s | 4.84s |
| Time to inference  | 47.31s | 55.27s| 24.33s |

