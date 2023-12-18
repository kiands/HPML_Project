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

Here we decide to choose two ways to do this special mapping task. The first way is classification using SVM or BERT. We will also try different ways to deploy the trained model and measured some metrics while running the BERT method. We compared the speed difference between CPU platforms and GPU platforms in inference tasks. We also tested the probability for running instances concurrently on CPU platforms and measured the memory bandwidth utilization and CPU utilization. We designed most of the experiments for inference task because it is more meaningful in the real world case: we cannot deploy deep learning models on devices with memory restrictions. This recognition service need to be provided as an API service.

The second way is to try large language models. We assume that the large language models have similar ability to recognize some specific patterns inside the strings we want to map. We may try to do the experiment at out of box status and also try fine-tuning. Due to the limited hardware resources, we will apply quantize, fine-tune with LoRA at the same time. Different bit depth while doing quantization will be considered. We will profile it using the nsys tool at the same time.
