# HPML_Project
## The model tuning and inference speed optimization for IoT vendor extraction
zh2553
hz3261

### Description
There are more and more devices with internet connection abilities in our daily life. They might be phones, tablets, routers, TVs and also smart IoT devices. For some reasons, users with technology background may be curious about the devices in their home intranet. As a result, tools like Fing were developed and a lot of device recognition databases were built.

At the beginning, those recognition methods were mostly based on the top 6 digits of MAC addresses. It is convenient for software to recognize at least the vendor of a device. However, at about 2019, Apple started to use a technology called random MAC address and many other companies started to use this.

Apparently this will neutralize the MAC database method. Then we need to find out a new way to do the recognition. Just like humans in real life, we leave trackable traces when we do various activities. Those devices who connect to the internet also leave traces.
Different devices from different vendors have very different traces.

For instance, a device might have a default name. A lot of such devices have very interesting default names that related to a unique vendor. Devices may also connect to its vendor’s server. Then we can sniff its internet connections and filter the domains. They may also transfer payloads of special protocols like so called netdisco. These information can be collected and combined with its real vendor to construct datasets.

Here is a sample data record for a smart lamp:
| name         | domain | netdisco                                   | vendor |
|--------------|--------|--------------------------------------------|--------|
| yeelight_ys21| null   | `{'vend':'XIAOMI Electronics,CO.,LTD'}`    | Xiaomi |

A lot of records are easy for human to find out the vendor’s name, but it is difficult to write a rule based recognition function because there are too many different conditions: loss of important data, capital words versus non capital words, full name versus simplified names, etc. If we need to map the relations of those vague but sufficient information with the very vendor name, machine learning is reasonable.

Here we decide to choose two ways to do this special mapping task. The first way is classification using SVM or BERT. We will also try different ways to deploy the trained model and measure some metrics while running the BERT method. We will compared the speed difference between CPU platforms and GPU platforms in inference tasks. We will also test the probability for running instances concurrently on CPU platforms and measure the memory bandwidth utilization and CPU utilization. We designed most of the experiments for inference task because it is more meaningful in the real world case: we cannot deploy deep learning models on devices with memory restrictions. This recognition service need to be provided as an API service.

The second way is to try large language models. We assume that the large language models have similar ability to recognize some specific patterns inside the strings we want to map. We may try to do the experiment at out of box status and also try fine-tuning. Due to the limited hardware resources, we will apply quantize, fine-tune with LoRA at the same time. Different bit depth while doing quantization will be considered. We will profile it using the nsys tool at the same time.

### Milestones
- (1) Accomplish good extraction accuracy on BERT model classification. (precision > 99% mostly) Completed
- (2) Check the propability of using SVM. Completed
- (3) Successfully adapted the model into onnx runtime and measure the performance. Completed
- (4) Run different types of deployment performance test: multi thread CPU, multi thread CPU ONNX, GPU, multi GPU DP, multi GPU DDP. Completed
- (5) Test the availability of llama-2 and try prompt engineering, measure the influence of prompts. Completed
- (6) Measure the influence of quantize. Completed
- (7) Test the fine-tune of llama-2 and try to find out the most important factor that influences the effectiveness. Completed

### Description of the repository and code structure
- `SVM`: data and code used to show the effect of SVM approach. Detailed code structure description is in the folder.
- `BERT_Approach`: data and code used to show the effect of BERT approach. Detailed code structure description is in the folder.
- `GGML`: data and code used to show the effect of GGML(llama.cpp) based LLM approach. Detailed code structure description is in the folder.
- `Huggingface`: data and code used to show the effect of Huggingface based LLM approach, mainly fine-tune. Detailed code structure description is in the folder.

### Example commands to execute the code
Similar to the previous section, the example commands are written in the readme of each folders (SVM, BERT_Approach, GGML, Huggingface). Enter the folder to access.

### Results (including charts/tables) and your observations
System-local: 13900K, RTX 4080 with 80GB/s memory bandwidth; System-HPC: 4*V100
1. SVM Precision compared to BERT (Small example)

|          | SVM with TFIDF | BERT   |
|----------|----------------|--------|
| Apple    | 0.93           | 0.99966|
| Google   | 0.83           | 0.998  |
| Amazon   | 0.97           | 0.986  |
| Microsoft| 0.73           | 1      |
2. BERT experiments and profiling results - CPU

| Configuration         | Model Load Time (s) | Inference Time (s) | Memory Bandwidth (GB/s) |
|-----------------------|---------------------|--------------------|-------------------------|
| CPU(Pytorch): 1 Thread| 0.61                | 6.15               | 26.71                   |
| CPU(Pytorch): 2 Threads| 0.62               | 4.89               | 35.99                   |
| CPU(Pytorch): 4 Threads| 0.67               | 4.06               | 50.32                   |
| CPU(ONNX): 1 Thread   | 0.36                | 9.25               | 17.97                   |
| CPU(ONNX): 2 Threads  | 0.45                | 5.00               | 11.48                   |
| CPU(ONNX): 4 Threads  | 0.50                | 24.10              | 7.67                    |
3. BERT experiments and profiling results - GPU

| GPU Configuration | Time to Load Model (s) | Time to Inference (s) |
|-------------------|------------------------|-----------------------|
| Single V100       | 2.97                   | 47.31                 |
| 4x V100 DP        | 3.09                   | 55.27                 |
| 4x V100 DDP       | 4.84                   | 24.33                 |
4. The time used to inference 20 records on different model-template combinations

|                   | 7B 4bit | 7B 8bit |
|-------------------|---------|---------|
| Detailed Template | 3.252s  | 10.887s |
| Vague Template    | 15.252s | 30.023s |
5. The influence on loss from different fine-tune templates
![llm-fine-tune-loss](./llm-fine-tune-loss.png)

Observations:
1. SVM always has lower accuracy than BERT. For example, its accuracy for Apple label prediction is 0.93 while BERT is 0.9996.
2. BERT runs not so slow on CPU, 4 threads with 100 records each on 13900K uses less than 5 seconds while RTX 4080 uses 2.1 seconds. V100 and multi V100 are very slow comparatively.
3. LLAMA needs to have good prompt engineering to make it output short and formatted result. Or it is nearly useless.
4. 4-bit quantize sometimes reduces almost 80% of the time when compared with 8-bit quantize.
5. Quantize itself takes some time to convert the data but it is useful.
6. Format of fine-tune data influences the fine-tune loss far more than other hyperparameters.
7. BERT is good. In many classes, it can have nearly 100% accuracy. Its speed is fast enough, 4 concurrent model instances on modern CPUs without memory bandwidth is as good as GPU inference. The other boosting methods may decrease performance. However they may have better compatibility. The future model inference deployment might have more and more accelerators other than GPU.
8. Truncated LLMs are not good in those tasks. Performances are restricted and we need to use a lot more tokens to justify its output. If we have other smaller models, they might be able to run faster and do better.
