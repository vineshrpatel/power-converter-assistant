from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from sentence_transformers import SentenceTransformer
import torch
import random
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import numpy as np

# Step 1. Chunk Documents
EMBEDDINGS_FILE = "embeddings.npz"
CONTENT_FILE = "content.json"

# Compute embeddings only if not saved
if not os.path.exists(EMBEDDINGS_FILE):
    text_splitter = TokenTextSplitter(chunk_size=1024, chunk_overlap=0)
    loader = PyPDFLoader("final_dataset_source.pdf") # dataset source pdf
    raw_chunks = loader.load_and_split(text_splitter)

    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)

    model_name = "all-mpnet-base-v2"
    embedding_model = SentenceTransformer(model_name, device='cuda')

    content = [rc.page_content for rc in raw_chunks]  # Extract text
    with torch.no_grad():
        embeddings = embedding_model.encode(content, batch_size=32)

    np.savez_compressed(EMBEDDINGS_FILE, embeddings=embeddings)
    with open(CONTENT_FILE, "w") as f:
        json.dump(content, f)


def load_embeddings():
    """Load embeddings from file if available."""
    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(CONTENT_FILE):
        print("Using existing embeddings")
        data = np.load(EMBEDDINGS_FILE)
        with open(CONTENT_FILE, "r") as f:
            content = json.load(f)
        return data["embeddings"], content
    else:
        raise FileNotFoundError("Embeddings file not found! Run embedding generation first.")

# Load embeddings and content
embeddings, content = load_embeddings()


# Step 2: Generate context by selecting chunks
REFERENCE_INDEX_FILE = "reference_index.json"
SIMILAR_INDICES_FILE = "similar_indices.json"
PROCESSED_CONTEXTS_FILE = "processed_contexts.json"

def get_reference_index(embeddings):
    """Retrieve the saved reference index or generate a new one if missing."""
    if os.path.exists(REFERENCE_INDEX_FILE):
        with open(REFERENCE_INDEX_FILE, "r") as f:
            ref_data = json.load(f)
            return ref_data["ref_index"]
    else:
        # reference_index = random.randint(0, len(embeddings) - 1)
        avg_embedding = np.mean(embeddings, axis=0)
        best_index = np.argmax([np.dot(emb, avg_embedding) / (np.linalg.norm(emb) * np.linalg.norm(avg_embedding)) for emb in embeddings])
        reference_index = best_index
        print(reference_index)
        with open(REFERENCE_INDEX_FILE, "w") as f:
            json.dump({"ref_index": int(reference_index)}, f)
        return reference_index

def get_similar_indices(embeddings, reference_embedding, similarity_threshold=0.4):
    """Retrieve or compute similar chunk indices based on the reference embedding."""
    if os.path.exists(SIMILAR_INDICES_FILE):
        with open(SIMILAR_INDICES_FILE, "r") as f:
            data = json.load(f)
            return data["indices"]
    else:
        similar_indices = []
        for i, embedding in enumerate(embeddings):
            product = np.dot(reference_embedding, embedding)
            norm = np.linalg.norm(reference_embedding) * np.linalg.norm(embedding)
            similarity = product / norm
            if similarity >= similarity_threshold:
                similar_indices.append(i)

        with open(SIMILAR_INDICES_FILE, "w") as f:
            json.dump({"indices": similar_indices}, f)
        return similar_indices


# Step 3: Load LLM
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# Set up the TinyLlama pipeline
pipe = pipeline("text-generation", model="microsoft/Phi-3-mini-4k-instruct", torch_dtype=torch.bfloat16, device_map="auto")

# Evolution prompt templates as strings
multi_context_template = """
Rewrite the `input` in under 15 words using information from the `Context`.
Ensure it remains a question and is fully answerable from 'Context' but is not dependent on 'Context' in order to understand it.
Do not use phrases like 'based on the provided context.'
Context: {context}
Input: {original_input}
Rewritten Input:"""

reasoning_template = """
Rewrite the given `input` so that it explicitly requests multi-step reasoning.
It should remain a question but should require multiple logical connections, should not exceed 15 words and be fully answerable from 'Context'.
Do not use phrases like 'based on the provided context' and ensure it is not dependent on 'Context' in order to understand it.
Context: {context}
Input: {original_input}
Rewritten Input:"""

evolution_templates = [multi_context_template, reasoning_template]

def generate_query(context):
    message = [
        {"role": "system", "content": "Generate a question about a power converter topology, ensuring it can be answered by the context but makes sense without having seen the context."},
        {"role": "user", "content": context},
    ]
    generation_args = {
        "max_new_tokens": 256,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }
    output = pipe(message, **generation_args)

    question = output[0]['generated_text']

    return question

def evolve_query(original_input, context, steps):
    current_input = original_input
    for _ in range(steps):
        chosen_template = random.choice(evolution_templates)
        evolved_prompt = chosen_template.format(context=context, original_input=current_input)

        messages = [
            {"role": "system", "content": "Improve the initial query."},
            {"role": "user", "content": evolved_prompt}
        ]
        generation_args = {
            "max_new_tokens": 256,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }
        outputs = pipe(messages, **generation_args)
        current_input = outputs[0]['generated_text'].split("Rewritten Input:")[-1].strip()
    return current_input

def generate_answer(context, query):
    messages = [
        {"role": "system", "content": "Answer the question about power converters based on the given context in a concise but complete manner. Ensure the answer does not cut off mid-sentence."},
        {"role": "user", "content": f"Given this context: {context}. Answer the question concisely: {query}"},
    ]
    generation_args = {
        "max_new_tokens": 512,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }
    outputs = pipe(messages, **generation_args)
    return outputs[0]['generated_text']

# Main process
def get_processed_indices():
    """Retrieve processed context indices from file."""
    if os.path.exists(PROCESSED_CONTEXTS_FILE):
        with open(PROCESSED_CONTEXTS_FILE, "r") as f:
            return set(json.load(f)["processed_indices"])
    return set()

def update_processed_indices(index):
    """Append processed index to file."""
    processed_indices = get_processed_indices()
    processed_indices.add(index)

    with open(PROCESSED_CONTEXTS_FILE, "w") as f:
        json.dump({"processed_indices": list(processed_indices)}, f)

def process_contexts_in_batches(evolution_steps=3, save_path="final_dataset.json"):
    """Processes contexts incrementally and saves progress."""
    
    # Load embeddings, content, reference index, and similar indices
    embeddings, content = load_embeddings() # saved
    reference_index = get_reference_index(embeddings) # saved
    reference_embedding = embeddings[reference_index]
    similar_indices = get_similar_indices(embeddings, reference_embedding) # saved
    
    # Track processed contexts
    processed_indices = get_processed_indices()

    # Load existing results if available
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            existing_results = json.load(f)
    else:
        existing_results = []

    for i in similar_indices:
        if i in processed_indices:
            continue  # Skip already processed contexts

        context = content[i]
        
        try:
            # Generate initial query
            initial_query = generate_query(context)
            
            # Evolve the query
            evolved_query = evolve_query(initial_query, context, steps=evolution_steps)
            
            # Generate answer
            answer = generate_answer(context, evolved_query)

            # Append and save result
            new_entry = {"evolved_query": evolved_query, "answer": answer}
            existing_results.append(new_entry)

            with open(save_path, "w") as f:
                json.dump(existing_results, f, indent=4)

            # Mark context as processed
            update_processed_indices(i)

            print(f"Processed context {i}/{len(similar_indices)}")

        except Exception as e:
            print(f"Error processing context {i}: {e}")

# Run processing
process_contexts_in_batches()