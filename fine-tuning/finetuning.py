import GPUtil
import platform

# Retrieve CPU and GPU model
cpu_model = platform.processor()
gpus = GPUtil.getGPUs()
gpu_model = gpus[0].name if gpus else "No GPU detected"
print(f"CPU Model: {cpu_model}")
print(f"GPU Model: {gpu_model}")

import wandb
wandb.login(key="<your_wandb_key>")

import json
import pandas as pd

# Opening JSON file
with open('final_dataset.json', 'r') as f:
    data = json.load(f)  # Load JSON data

data_df = pd.DataFrame(data)

from sklearn.model_selection import train_test_split

# Perform train-test split
train, test = train_test_split(data_df, test_size=0.3, random_state=42)
train, validation = train_test_split(train, test_size=0.2, random_state=42)

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

######################################################################################
######################################################################################
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
    lora_alpha=16,
    lora_dropout=0.5,
    r=128,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear"
)

# merges base model + LoRA adapters
pretrained_model = get_peft_model(base_model, peft_config) 

# Initialize wandb
wandb.init(
    project="capstone-finetuning",  # your project name
    name="finetuning-sweep2",  # name of this specific run
)

from transformers import TrainingArguments

training_arguments = TrainingArguments(
    output_dir="./ncc-finetuned_finaldataset",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    optim="paged_adamw_32bit",
    num_train_epochs=5,
    learning_rate=1e-5,
    fp16=True, # GPU usage
    max_grad_norm=1,
    warmup_ratio=0.1,
    group_by_length=True,
    lr_scheduler_type="linear",
    save_strategy="epoch", # save model every epoch
    logging_strategy="epoch", # log metrics every epoch
    evaluation_strategy="epoch", # evaluate every epoch
    report_to="wandb"
)

from trl import SFTConfig

sft_config = SFTConfig(
    output_dir="./ncc-finetuned_finaldataset",
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

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
print_trainable_parameters(trainer.model)

# Start training
trainer.train()
# # Resume training if crashed
# trainer.train(resume_from_checkpoint=True)

# Evaluate the model
eval_results = trainer.evaluate()

# Log the evaluation results
wandb.log(eval_results)

wandb.finish()

print("Training Complete")

# Save model
model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  # Take care of distributed/parallel training
model_to_save.save_pretrained("finetuned_output_incomplete")

lora_config = LoraConfig.from_pretrained('finetuned_output_incomplete')

# Step 1: Merge the LoRA weights with the base model
merged_model = pretrained_model.merge_and_unload()

# Step 2: Save the entire merged model
merged_model.save_pretrained("complete_fine_tuned_model")

# Step 3: Save the tokenizer
trainer.tokenizer.save_pretrained("complete_fine_tuned_model")

print("Complete fine-tuned model and tokenizer saved to 'complete_fine_tuned_model'")
