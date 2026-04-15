import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    BertTokenizerFast, 
    BertForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score
import torch

# AG News has 4 classes (1-indexed in CSV, we convert to 0-indexed)
LABELS = ["World", "Sports", "Business", "Sci/Tech"]
NUM_LABELS = 4

# Load Data
train_df = pd.read_csv("data/train.csv", header=None, names=["label", "title", "description"])
test_df  = pd.read_csv("data/test.csv",  header=None, names=["label", "title", "description"])

# Combine title + description as input text; convert labels to 0-indexed
train_df["text"] = train_df["title"] + " " + train_df["description"]
test_df["text"]  = test_df["title"]  + " " + test_df["description"]
train_df["label"] = train_df["label"] - 1
test_df["label"]  = test_df["label"]  - 1

# Tokenize
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

train_dataset = Dataset.from_pandas(train_df[["text", "label"]])
test_dataset  = Dataset.from_pandas(test_df[["text", "label"]])

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset  = test_dataset.map(tokenize, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch",  columns=["input_ids", "attention_mask", "label"])

# Model 
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels=NUM_LABELS
)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

# Training
# Reduce num_train_epochs to 1-2 if running on CPU
training_args = TrainingArguments(
    output_dir="./bert-news-model",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("./bert-news-model/final")
print("Training complete. Model saved.")