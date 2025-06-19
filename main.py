import os
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
import evaluate
import numpy as np
from transformers import TrainingArguments
from transformers import Trainer
import dotenv

dotenv.load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
MOUNT_PATH = os.getenv("MOUNT_PATH")


raw_datasets = load_dataset("glue", "mrpc", cache_dir=os.path.join(MOUNT_PATH, "datasets"), token=HF_TOKEN)
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=os.path.join(MOUNT_PATH, "models"), token=HF_TOKEN)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2, cache_dir=os.path.join(MOUNT_PATH, "models"), token=HF_TOKEN)

optimizer = AdamW(model.parameters(), lr=5e-5)
metric = evaluate.load("glue", "mrpc", cache_dir="./cache")


def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    "./cache", 
    max_steps=5, # for testing
    eval_strategy="epoch",
    # Enable GPU usage
    no_cuda=False,  # This enables CUDA when available
    dataloader_pin_memory=True,  # Speeds up data loading on GPU
)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,  # Fixed: was processing_class, should be tokenizer
    compute_metrics=compute_metrics,
)


print("Device being used:", trainer.args.device)


trainer.train()


