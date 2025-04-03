import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json
import os

# change this to desired model
base_model_name = "microsoft/Phi-3-mini-4k-instruct" 

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    trust_remote_code=True
)
base_model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.padding_side = 'right'
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    base_model.resize_token_embeddings(len(tokenizer))


########################################################################
########################################################################
# Load test portion of dataset (save this as a json file)
with open('test_dataset.json') as f:
    results_data = json.load(f)

# Check for existing output file
output_file = 'basemodel_generation.json' # for fine-tuned model change to finetuned_generation.json 
processed_queries = set()

if os.path.exists(output_file):
    with open(output_file, 'r') as f:
        try:
            existing_data = json.load(f)
            processed_queries = {entry['evolved_query'] for entry in existing_data}
        except json.JSONDecodeError:
            existing_data = []
else:
    existing_data = []

output_data = existing_data  # Continue appending to existing data

for i in results_data:
    text = i['evolved_query']
    ref_ans = i['answer']

    # Skip already processed queries
    if text in processed_queries:
        continue

    try:
        # Generate response
        messages = [
            {"role": "user", "content": text},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt").to('cuda')

        # Generate Response
        outputs = base_model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
        )
        decoded_text = tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):], 
            skip_special_tokens=True
        )

        # Store result
        result_dict = {
            'evolved_query': text,
            'ref_ans': ref_ans,
            'generated_text': decoded_text,
            'rubric': "criteria: Does the response answer the instruction in a concise and accurate way?, score1_description: The response is not able to give a clear answer to the instruction, score2_description: The response is vague but has somewhat relevant details about power converters, score3_description: The response is generally ok and provides some detail about power converters, score4_description: The response answers the instruction and provides relevant detail about power converters and their design in an engineering context, score5_description: The response clearly and concisely answers the question, providing relevant detail about the power converter topology under consideration in an engineering context."
        }
        output_data.append(result_dict)

        # Save progress after each entry
        with open(output_file, 'w') as outfile:
            json.dump(output_data, outfile, indent=4)

        # Mark as processed
        processed_queries.add(text)

    except Exception as e:
        print(f"Error processing query: {text}\nError: {str(e)}")
        continue  # Move on to the next query

print("Processing complete!")