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
# 0. IMPORTS
# ================================================
import json
import pandas as pd
from pandas import json_normalize
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

from sklearn.metrics import accuracy_score, f1_score
import numpy as np


# ================================================
# 1. LOAD + CLEAN JSONL DATA
# ================================================
def load_jsonl(path):
    data_list = []
    with open(path, "r") as f:
        for line in f:
            try:
                data_list.append(json.loads(line))
            except json.JSONDecodeError as e:
                print("JSON error:", e)
    return json_normalize(data_list)


train_data = load_jsonl("train.jsonl")
kaggle_data = load_jsonl("kaggle_test.jsonl")


# ================================================
# 2. EXTRACT FULL TEXT (EXTENDED TWEETS)
# ================================================
def extract_full_text(row):
    if "extended_tweet.full_text" in row and not pd.isna(row["extended_tweet.full_text"]):
        return row["extended_tweet.full_text"]
    return row.get("text", "")

train_data["full_text"] = train_data.apply(extract_full_text, axis=1)
kaggle_data["full_text"] = kaggle_data.apply(extract_full_text, axis=1)


# ============================================================
# 3. BUILD HUGGINGFACE DATASETS
# ============================================================
raw_train_dataset = Dataset.from_pandas(train_data[["full_text", "label"]])
raw_kaggle_dataset = Dataset.from_pandas(kaggle_data[["full_text"]])

# Convert integer labels → ClassLabel (for stratification)
from datasets import ClassLabel
class_label = ClassLabel(num_classes=2, names=["observer", "influencer"])
raw_train_dataset = raw_train_dataset.cast_column("label", class_label)

# Stratified train/val split
raw_split = raw_train_dataset.train_test_split(
    test_size=0.1,
    stratify_by_column="label"
)

raw_train_dataset = raw_split["train"]
raw_val_dataset   = raw_split["test"]


# ================================================
# 4. TOKENIZATION
# ================================================
model_name = "nreimers/MiniLM-L6-H384-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_fn(batch):
    return tokenizer(
        batch["full_text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

ds_train = raw_train_dataset.map(tokenize_fn, batched=True, load_from_cache_file=False)
ds_val   = raw_val_dataset.map(tokenize_fn, batched=True, load_from_cache_file=False)

# Trainer expects "labels"
ds_train = ds_train.rename_column("label", "labels")

ds_train.set_format("torch")
ds_val.set_format("torch")


# ================================================
# 5. METRICS
# ================================================
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }


# ================================================
# 6. TRAIN MINILM (COLAB GPU-OPTIMIZED)
# ================================================
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

training_args = TrainingArguments(
    output_dir="minilm_final",
    report_to="none",
    learning_rate=3e-5,
    per_device_train_batch_size=32,   # larger batch fits on GPU
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs_minilm",

    # 🚀 GPU acceleration
    fp16=True,          # fastest on T4/V100/A100
    bf16=False,         # NVIDIA fp16 preferred
    no_cuda=False,      # ensures GPU is used
    torch_compile=True, # PyTorch 2.0 speed boost

    save_total_limit=1,
    eval_strategy="epoch",
    logging_strategy="steps",
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_train,
    eval_dataset=ds_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("🚀 Starting MiniLM training on GPU...")
trainer.train()
print("✅ Training finished.")


# ================================================
# 7. SAVE TRAINED MODEL
# ================================================
trainer.save_model("minilm_final")
tokenizer.save_pretrained("minilm_final")
print("📦 Model saved in minilm_final/")


# ================================================
# 8. GENERATE KAGGLE PREDICTIONS
# ================================================
ds_kaggle = raw_kaggle_dataset.map(
    tokenize_fn,
    batched=True,
    load_from_cache_file=False
)
ds_kaggle.set_format("torch")

preds = trainer.predict(ds_kaggle).predictions
pred_labels = preds.argmax(axis=1)

submission = pd.DataFrame({
    "id": kaggle_data["id"],
    "label": pred_labels
})

submission.to_csv("submission.csv", index=False)
print("📄 Saved submission.csv")