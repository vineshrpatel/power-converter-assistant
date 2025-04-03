# power-converter-assistant
Python scripts for the development of an LLM-based framework to aid with power converter design. Includes a Flask web application for running the framework.

## dataset
```synthetic_data.py``` generates a question-answer (QA) pair dataset based on a PDF using Microsoft's Phi-3-Mini-4K-Instruct LLM.

```final_dataset.json``` contains the full QA pair dataset and ```test_datase.json``` contains the test split.

## hyperparameter-sweep
```hyperparameter_optimisation.py``` performs a hyperparameter sweep using Bayesian optimisation for fine-tuning Phi-3-Mini-4K-Instruct using QLoRA. Data is logged to Weights & Biases (https://wandb.ai/site) using your API key.

## fine-tuning
```finetuning.py``` fine-tunes Phi-3-Mini-4K-Instruct for set hyperparameters and logs metrics to Weights & Biases. The fine-tuned model is saved locally after completion.

## testing
Run ```prometheus_data.py``` first to obtain answers from the model you want to test.<br>

```prometheus_testing.py``` scores the model using the Prometheus-Eval framework (https://github.com/prometheus-eval/prometheus-eval). The script is dependent on the file produced from ```prometheus_data.py```.<br>

```f1pendant_scores.py``` scores the model using the PENDANT similarity index from qa-metrics (https://github.com/zli12321/qa_metrics). The script is also dependent on the file produced from ```prometheus_data.py```