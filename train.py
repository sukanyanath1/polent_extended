import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding,EarlyStoppingCallback
from datasets import DatasetDict, Dataset
import torch
import numpy as np
import evaluate
import logging
from sklearn.metrics import confusion_matrix as cm
import time
import json
import os
import argparse

formatter = logging.Formatter('%(asctime)s %(lineno)s %(levelname)s %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument("--model_saving_path", default=os.getenv("MODEL_SAVING_PATH"))
parser.add_argument("--results_path", default=os.getenv("RESULTS_PATH"))
parser.add_argument("--model_name", default=os.getenv("MODEL_NAME", "bert-base-uncased"))
parser.add_argument("--dataset_path", default=os.getenv("DATASET_PATH"))
args, _ = parser.parse_known_args()

MODEL_NAME = args.model_name
MODEL_SAVING_PATH = os.path.join(args.model_saving_path, MODEL_NAME)
RESULTS_PATH = os.path.join(args.results_path, MODEL_NAME, "results.json")
DATASET_PATH = args.dataset_path
LOG_DIR = os.path.join(MODEL_SAVING_PATH, "logs")
LOG_FILE = os.path.join(LOG_DIR, "train.log")

if not args.model_saving_path or not args.results_path or not DATASET_PATH:
    raise ValueError(
        "Missing required paths. Provide --model_saving_path, --results_path, and --dataset_path "
        "or set MODEL_SAVING_PATH, RESULTS_PATH, and DATASET_PATH environment variables."
    )

os.makedirs(MODEL_SAVING_PATH, exist_ok=True)
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def setup_logger(name, log_file=LOG_FILE, level=logging.INFO):
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.addHandler(handler)
    return logger

def tokenize_function(batch, max_length):
    tokenized_batch = tokenizer(
        text=batch["sent"],
        text_pair=batch["head_tail"],
        padding=True,
        max_length=max_length,
        truncation=True,
    )
    tokenized_batch["labels"] = batch[LABEL_COLUMN]
    return tokenized_batch

logger = setup_logger("logger")
logger.info(f"Model saving path: {MODEL_SAVING_PATH}")
logger.info(f"Results path: {RESULTS_PATH}")

logger.info(f'Is Cuda available? {torch.cuda.is_available()}')

data = DatasetDict.load_from_disk(DATASET_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_columns = data["train"].column_names
LABEL_COLUMN = next((c for c in ["labels", "label", "Park"] if c in train_columns), None)
if LABEL_COLUMN is None:
    raise ValueError(f"Could not find label column in train split. Available columns: {train_columns}")

num_labels = len(set(data["train"][LABEL_COLUMN]))
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
original_columns = data["train"].column_names
tokenized_datasets = data.map(
    tokenize_function,
    batched=True,
    fn_kwargs={"max_length": tokenizer.model_max_length},
    remove_columns=original_columns,
)

# Define training arguments
training_args = TrainingArguments(
    output_dir=MODEL_SAVING_PATH,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir=LOG_DIR,
    logging_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    greater_is_better=True,
    report_to="none"
)


accuracy = evaluate.load("accuracy")
# Define compute metrics funct

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    labels = labels.squeeze()
    return accuracy.compute(predictions=predictions, references=labels)

#4. Define data_collators
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding = True) #pads to the max sequence length in a batch

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator = data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] 
)

logger.info(f'Training started')
start = time.time()
# Train model
trainer.train()
logger.info(f"Execution Time: {(time.time()-start):.2f} seconds")

# Evaluate model
valid_results = trainer.evaluate(tokenized_datasets["valid"])
logger.info(f"Validation Accuracy: {valid_results['eval_accuracy']:.4f}")
test_results = trainer.evaluate(tokenized_datasets["test"])
logger.info(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")

predictions = trainer.predict(test_dataset=tokenized_datasets["test"])

# Extract the predicted labels and the true labels
predicted_labels = predictions.predictions.argmax(axis=-1)
true_labels = predictions.label_ids

# Save the results
results = {
    "Validation Accuracy": valid_results["eval_accuracy"],
    "Test Accuracy": test_results["eval_accuracy"],
    "Predicted Labels": predicted_labels.tolist(),
    "True Labels": true_labels.tolist()
}
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f)