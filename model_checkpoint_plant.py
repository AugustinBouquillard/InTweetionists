"""from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Load tiny model
model = SentenceTransformer("sentence-transformers/paraphrase-albert-small-v2")

# Prepare your data
texts = train_data["full_text"].tolist()
labels = train_data["label"].values

# Compute embeddings (VERY fast)
embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)

# Split
X_train, X_val, y_train, y_val = train_test_split(embeddings, labels, test_size=0.1)

# Train a classifier
clf = LogisticRegression(max_iter=2000)
clf.fit(X_train, y_train)

# Evaluate
acc = clf.score(X_val, y_val)
print("Validation accuracy:", acc)"""



# ================================================
# 0. IMPORTS & GPU CHECK
# ================================================
import os
import json
import numpy as np
import pandas as pd
from pandas import json_normalize
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score, f1_score
import torch

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU device:", torch.cuda.get_device_name(0))
else:
    raise SystemError("GPU NOT detected. In Colab, go to Runtime → Change Runtime Type → GPU.")

device = torch.device("cuda")


# ================================================
# 1. LOAD JSONL DATA
# ================================================
def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except:
                pass
    return json_normalize(rows)

train_data = load_jsonl("train.jsonl")
kaggle_data = load_jsonl("kaggle_test.jsonl")

print("Train rows:", len(train_data))
print("Kaggle rows:", len(kaggle_data))


# ================================================
# 2. EXTRACT FULL TEXT
# ================================================
def extract_full_text(row):
    if "extended_tweet.full_text" in row and not pd.isna(row["extended_tweet.full_text"]):
        return row["extended_tweet.full_text"]
    return row.get("text", "")

train_data["full_text"]  = train_data.apply(extract_full_text, axis=1)
kaggle_data["full_text"] = kaggle_data.apply(extract_full_text, axis=1)


# ================================================
# 3. BUILD HF DATASETS
# ================================================
raw_train = Dataset.from_pandas(train_data[["full_text", "label"]])
raw_test  = Dataset.from_pandas(kaggle_data[["full_text"]])

from datasets import ClassLabel
labels = ClassLabel(num_classes=2, names=["observer", "influencer"])
raw_train = raw_train.cast_column("label", labels)

split = raw_train.train_test_split(test_size=0.1, stratify_by_column="label")
ds_train = split["train"]
ds_val   = split["test"]

print("Train:", len(ds_train), " Val:", len(ds_val))


# ================================================
# 4. TOKENIZATION (Dynamic padding)
# ================================================
MODEL_NAME = "nreimers/MiniLM-L6-H384-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["full_text"], truncation=True, max_length=128)

ds_train = ds_train.map(tokenize, batched=True)
ds_val   = ds_val.map(tokenize, batched=True)
ds_test  = raw_test.map(tokenize, batched=True)

ds_train = ds_train.rename_column("label", "labels")
ds_train.set_format("torch")
ds_val.set_format("torch")
ds_test.set_format("torch")

collator = DataCollatorWithPadding(tokenizer=tokenizer)


# ================================================
# 5. METRICS
# ================================================
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }


# ================================================
# 6. LOAD BASE MODEL + LoRA
# ================================================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=2
)

# LoRA config optimized for MiniLM
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query", "key", "value", "dense"],  # works for MiniLM
    lora_dropout=0.05,
    task_type=TaskType.SEQ_CLS
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ================================================
# 7. TRAINING ARGS (GPU + FP16 + EARLY STOPPING)
# ================================================
training_args = TrainingArguments(
    output_dir="lora_minilm_out",
    learning_rate=2e-4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=5,                     # early stopping will stop earlier
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,

    fp16=True,                              # use GPU fast path
    dataloader_num_workers=4,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_train,
    eval_dataset=ds_val,
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(
        early_stopping_patience=2,          # stops if 2 consecutive epochs have no improvement
        early_stopping_threshold=0.0001
    )]
)


# ================================================
# 8. TRAIN
# ================================================
print("🚀 Training with LoRA on GPU...")
trainer.train()
print("✅ Training complete!")


# ================================================
# 9. SAVE PEFT ADAPTER
# ================================================
model.save_pretrained("minilm_lora_adapter")
tokenizer.save_pretrained("minilm_lora_adapter")
print("📦 Saved LoRA adapter → minilm_lora_adapter/")


# ================================================
# 10. PREDICT TEST SET
# ================================================
preds = trainer.predict(ds_test).predictions
labels = preds.argmax(axis=1)

submission = pd.DataFrame({
    "ID": kaggle_data["challenge_id"],
    "Prediction": labels
})
submission.to_csv("submission.csv", index=False)

print("📄 Saved submission.csv")