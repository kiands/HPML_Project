# Model: llama-2

## Huggineface:
- `llama-2-7b-hf`: folder to store huggingface format 7B llama-2, can be downloaded via huggingface.co
- `llama-2-7b-hf-fine-tune-baby`: folder to store fine tuned adapters of llama-2 model
- `fine-tune-llama2.ipynb`: manually created fine tune program using huggingface libraries
- `cs_train.jsonl`: training data for fine-tune-llama2.ipynb
- `cs_test.jsonl`: testing data for fine-tune-llama2.ipynb
- `cs_train_new.json`: converted dataset in another format to fit for LLaMA-Factory toolkit
- `dataset_info.json`: converted description data to fit for LLaMA-Factory toolkit

## Commands (at the root of GGML folder):
1. Install requirements:
python3 -m pip install -r requirements.txt
2. Fine tune the model using the notebook (make sure model has been downloaded)
3. Clone the LLaMA-Factory repository:
git clone https://github.com/hiyouga/LLaMA-Factory.git
4. Enter LLaMA-Factory:
cd LLaMA-Factory
5. Execute the command (make sure the path is correctly written):
### Qlora
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path ../llama-2-7b-hf \
    --do_train True \
    --finetuning_type lora \
    --quantization_bit 4 \
    --template default \
    --flash_attn False \
    --shift_attn False \
    --dataset_dir .. \
    --dataset cs_train_new \
    --cutoff_len 1024 \
    --learning_rate 5e-05 \
    --num_train_epochs 1.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 100 \
    --save_steps 200 \
    --warmup_steps 0 \
    --neft_alpha 0 \
    --train_on_prompt False \
    --upcast_layernorm True \
    --lora_rank 8 \
    --lora_dropout 0.1 \
    --lora_target q_proj,v_proj \
    --resume_lora_training True \
    --output_dir lora \
    --fp16 True \
    --plot_loss True 

This experiment shows that the loss is significantly influenced by fine tune templates. Other hyper parameters are not very important.
