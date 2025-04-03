import wandb
wandb.login(key="<your_key>")

import json
import pandas as pd

# Opening JSON file
with open('final_dataset.json', 'r') as f1:
    data = json.load(f1)  # Load JSON data

data_df = pd.DataFrame(data)

from sklearn.model_selection import train_test_split    

# Perform train-test split
train, test = train_test_split(data_df, test_size=0.3, random_state=42)
train, validation = train_test_split(train, test_size=0.2, random_state=42)

# Save test data for model testing scripts
test_formatted = []
for _, row in test.iterrows():
    test_formatted.append({
        "evolved_query": row['evolved_query'],
        "answer": row['answer']
    })
with open('test_dataset.json', 'w') as outfile:
    json.dump(test_formatted, outfile, indent=4)

import datasets

def format_example(example):
    """Formats the example with clear separators."""
    text = f""
    if example['evolved_query']:
        text += f"Input:\n{example['evolved_query']}\n"
    text += f"Output:\n{example['answer']}"
    example['text'] = text  # Create a new 'text' field
    return example

# Convert the Pandas DataFrame to a Hugging Face Dataset
train_data = datasets.Dataset.from_pandas(train)
train_data = train_data.map(format_example)

validation_data = datasets.Dataset.from_pandas(validation)
validation_data = validation_data.map(format_example)

print('Training data size:', len(train_data))

###########################################################
###########################################################

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def train_model():
    run = wandb.init()
    config = wandb.config

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

    from peft import LoraConfig, get_peft_model

    peft_config = LoraConfig(
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        r=config.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear"
    )

    # merges base model + LoRA adapters
    pretrained_model = get_peft_model(base_model, peft_config) 

    from transformers import TrainingArguments

    training_arguments = TrainingArguments(
        output_dir="./ncc-finetuned",
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        optim="paged_adamw_32bit",
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        fp16=True, # GPU usage
        max_grad_norm=config.max_grad_norm,
        warmup_ratio=config.warmup_ratio,
        group_by_length=True,
        lr_scheduler_type="linear",
        save_strategy="epoch",
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss"
    )

    from trl import SFTConfig

    sft_config = SFTConfig(
        output_dir="./ncc-finetuned",
        max_seq_length=1024
    )

    from trl import SFTTrainer

    trainer = SFTTrainer(
        model=pretrained_model,
        args=training_arguments,
        train_dataset=train_data,
        eval_dataset=validation_data,
        tokenizer=tokenizer,
        dataset_text_field="text"
    )

    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)
        
    # Start training
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()

    # Log the evaluation results
    wandb.log(eval_results)

sweep_config = {
    'method': 'bayes',
    'name': 'sweep_bayes_memory_test',
    'metric': {
        'name': 'eval_loss',
        'goal': 'minimize',
    },
    'parameters': {
        'lora_alpha': {'values': [16, 32, 64]},
        'lora_dropout': {'values': [ 0.1, 0.2, 0.5]},
        'lora_r': {'values': [32, 64, 128]},
        'batch_size': {'values': [1, 2, 4]},
        'gradient_accumulation_steps': {'values': [2, 4, 8]},
        'learning_rate': {'values': [1e-5, 5e-5, 1e-4, 1e-3]},
        'epochs': {'min': 3, 'max': 10},
        'max_grad_norm': {'values': [0.3, 1.0, 2.0]},
        'warmup_ratio': {'values': [0.03, 0.1, 0.2]}

    }
}

# Initialize the sweep (creates fresh sweep)
sweep_id = wandb.sweep(sweep_config, project="capstone-finetune-hyperparam")
wandb.agent(sweep_id, function=train_model, count=10)

# # If crashes during sweep find the sweep_id in url of sweep
# sweep_id = "<your_sweep_id>" 
# wandb.agent(sweep_id, function=train_model, project="capstone-finetune-hyperparam", count=10)

print("\nSweep Complete.")
