import pandas as pd
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate

loaded_df = pd.read_csv('../gpt_playaround_dataset.csv')

n_gpu_layers = -1  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
# Make sure the model path is correct for your system!
llm = LlamaCpp(
    # change model here
    model_path="../ggml-model-7b-q4_0.gguf",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    temperature=0.8,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
#     callback_manager=callback_manager,
    verbose=False,  # Verbose is required to pass to the callback manager,
)

# columns_to_copy is important. It should be aligned with the prompt templates. In this case, it is chat-with-bob-vendor-prediction-device_name.txt
columns_to_copy = ['device_id', 'netdisco_info']
df_llama2_device_name = loaded_df[loaded_df['netdisco_info'].notna()][columns_to_copy].copy()
# define number of records
topn = df_llama2_device_name.head(20)

# Choose the right template
with open('../chat-with-bob-vendor-prediction-netdisco.txt', 'r') as file:
    # Read the entire content of the file into a variable
    template_device_name = file.read()
    
prompt_device_name = template_device_name.rstrip() + " I have a device labeled as 'Cyndi's iPhone'. What is the vendor? \nBob: "

import time

start_time = time.time()

# Counter for printing messages
row_counter = 0

# Apply the function to each row
for index, row in topn.iterrows():
    # If this is device_name, we use row['device_name']. If others, we should make a change
    prompt_device_name = template_device_name.rstrip() + ' I have a device labeled as \'' + row['netdisco_info'] + '\'. What is the vendor? \nBob: '

    try:
        result = llm(prompt_device_name).strip().split('\n')[0].split('.')[0]
    except:
        result = 'fail_to_process'

    # Check this variable, too. Format: 'llama2_' + the current column name.
    df_llama2_device_name.at[index, 'llama2_netdisco_info'] = result

#     print(prompt) 
#     print(result)
    
    

    
    # Increment row counter
    row_counter += 1
    
    # Print message every 10 rows
    if row_counter % 10 == 0:
        print(f"Processed {row_counter} rows")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Done! Elapsed Time: {elapsed_time} seconds")
