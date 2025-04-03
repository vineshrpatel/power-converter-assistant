import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json
import os

model_name = "prometheus-eval/prometheus-7b-v2.0"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.padding_side = 'right'
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))


# for fine-tuned model change to finetuned_generation.json 
with open('basemodel_generation.json', 'r') as f:
    prometheus_input = json.load(f)

# Check for existing output file
output_file = 'prometheus_base.json' # for fine-tuned model change to prometheus_finetuned.json 
processed_queries = set()

if os.path.exists(output_file):
    with open(output_file, 'r') as f:
        try:
            existing_data = json.load(f)
            processed_queries = {entry['query'] for entry in existing_data}
        except json.JSONDecodeError:
            existing_data = []
else:
    existing_data = []

all_responses = existing_data

for i in prometheus_input:
    query = i['evolved_query']
    ans = i['ref_ans']
    gen = i['generated_text']
    eval_rubric = ['rubric']

    # Skip already processed queries
    if query in processed_queries:
        continue
    
    try:
        ABS_SYSTEM_PROMPT = "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."

        ABSOLUTE_PROMPT = """###Task Description:
        An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
        1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
        2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
        3. The output format must give the score first and then feedback: "[SCORE] (an integer number between 1 and 5)\nFeedback: (write a feedback for criteria)"
        4. Please do not generate any other opening, closing, and explanations.
        ###The instruction to evaluate:
            {query}

            ###Response to evaluate:
            {gen}

            ###Reference Answer (Score 5):
            {ans}

            ###Score Rubrics:
            {eval_rubric}

            ###Score: """

        evaluations = ABSOLUTE_PROMPT.format(
            query=query,
            gen=gen,
            ans=ans,
            eval_rubric=eval_rubric
        )

        messages = [
            {"role": "system", "content": ABS_SYSTEM_PROMPT},
            {"role": "user", "content": evaluations},
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True, # adds tokens to tell where start generating from
            return_dict=True, # returns attention tensors
            return_tensors="pt").to('cuda')

        # Generate Response
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

        # Decode Response
        decoded_text = tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True)
        
        # Store response in dictionary
        response_data = {
            "query": query,
            "reference_answer": ans,
            "generated_text": gen,
            "model_feedback": decoded_text
        }
        
        # Add to responses list
        all_responses.append(response_data)

        # Save progress after each entry
        with open(output_file, 'w') as outfile:
            json.dump(all_responses, outfile, indent=4)

        # Mark as processed
        processed_queries.add(query)

    except Exception as e:
        print(f"Error processing query: {query}\nError: {str(e)}")
        continue  # Move on to the next query

print("Processing complete!")