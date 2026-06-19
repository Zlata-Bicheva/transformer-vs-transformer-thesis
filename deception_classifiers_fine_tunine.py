# =====================================
# Combined Python Script
# Fine-tunes:
# 1. Decoder-only (GPT2)
# 2. Encoder-only (RoBERTa)
# 3. Encoder-Decoder (FLAN-T5 with CV)
# =====================================

import os
import re
import time
import numpy as np
import pandas as pd
import torch

from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, f1_score

# HF + datasets
from transformers import (
    GPT2TokenizerFast, GPT2LMHeadModel,
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

from torch.utils.data import Dataset, DataLoader
from torch import nn
from datasets import Dataset as HFDataset, DatasetDict

import kagglehub

# =====================================
# Global
# =====================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# =====================================
# DATA LOADING (shared)
# =====================================
def load_data():
    path = kagglehub.dataset_download("rtatman/deceptive-opinion-spam-corpus")
    csv_path = os.path.join(path, "deceptive-opinion.csv")

    df = pd.read_csv(csv_path)
    df = df[["text", "deceptive"]].dropna()
    df["text"] = df["text"].astype(str)
    return df

# =====================================
# 1. DECODER-ONLY (GPT2)
# =====================================
class GPT2Dataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx])
        }

class GPT2Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.gpt2.config.n_embd, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden = outputs.hidden_states[-1]
        lengths = attention_mask.sum(dim=1) - 1
        cls_repr = hidden[torch.arange(hidden.size(0)), lengths]
        logits = self.classifier(self.dropout(cls_repr))
        return logits


def train_decoder(df):
    print("\n=== Training GPT2 (Decoder-only) ===")

    df = df.copy()
    df["label"] = df["deceptive"].map({"truthful": 0, "deceptive": 1})

    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_loader = DataLoader(GPT2Dataset(train_df, tokenizer), batch_size=16, shuffle=True)
    valid_loader = DataLoader(GPT2Dataset(valid_df, tokenizer), batch_size=16)

    model = GPT2Discriminator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    def run_epoch(loader, train=True):
        model.train() if train else model.eval()
        total_loss, correct, total = 0, 0, 0

        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(ids, mask)
            loss = loss_fn(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        return total_loss / len(loader), correct / total

    for epoch in range(3):
        train_loss, train_acc = run_epoch(train_loader, True)
        val_loss, val_acc = run_epoch(valid_loader, False)
        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

    os.makedirs("./saved_models/decoder_only", exist_ok=True)
    torch.save(model.state_dict(), "./saved_models/decoder_only/pytorch_model.bin")
    tokenizer.save_pretrained("./saved_models/decoder_only")

# =====================================
# 2. ENCODER-ONLY (RoBERTa)
# =====================================
def train_encoder(df):
    print("\n=== Training RoBERTa (Encoder-only) ===")

    encoder_df = df.copy()
    encoder_df["label"] = encoder_df["deceptive"].replace({"truthful": 0, "deceptive": 1})

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        encoder_df["text"].tolist(),
        encoder_df["label"].tolist(),
        test_size=0.2,
        stratify=encoder_df["label"],
        random_state=42
    )

    MODEL_NAME = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(texts):
        return tokenizer(texts, padding="max_length", truncation=True, max_length=512)

    train_enc = tokenize(train_texts)
    test_enc = tokenize(test_texts)

    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = SimpleDataset(train_enc, train_labels)
    test_dataset = SimpleDataset(test_enc, test_labels)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./results/roberta",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=3,
            per_device_train_batch_size=16
        ),
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model("./saved_models/encoder_only_roberta")
    tokenizer.save_pretrained("./saved_models/encoder_only_roberta")

# =====================================
# 3. ENCODER-DECODER (FLAN-T5) EXACT VERSION
# =====================================

def preprocess(batch, tokenizer, max_input_len, max_label_len):
    inputs = [
        f"Classify the following hotel review as truthful or deceptive: {txt}"
        for txt in batch["text"]
    ]

    model_inputs = tokenizer(inputs, max_length=max_input_len, truncation=True, padding="max_length")
    labels = tokenizer(text_target=batch["label"], max_length=max_label_len, truncation=True, padding="max_length")

    labels_ids = [
        [(token if token != tokenizer.pad_token_id else -100) for token in seq]
        for seq in labels["input_ids"]
    ]

    model_inputs["labels"] = labels_ids
    return model_inputs


def compute_metrics_t5(eval_pred, tokenizer):
    preds, labels = eval_pred
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [p.strip().lower() for p in decoded_preds]
    decoded_labels = [l.strip().lower() for l in decoded_labels]

    y_pred = [1 if p == "deceptive" else 0 for p in decoded_preds]
    y_true = [1 if l == "deceptive" else 0 for l in decoded_labels]

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def train_flan(df):
    print("\n=== Training FLAN-T5 (Encoder-Decoder) ===")

    df = df.copy()
    df["label"] = df["deceptive"].str.strip().str.lower()

    MODEL_NAME = "google/flan-t5-base"
    MAX_INPUT_LEN = 256
    MAX_LABEL_LEN = 4

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df["text"], df["label"])):
        print(f"Fold {fold+1}")

        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        dataset = DatasetDict({
            "train": HFDataset.from_pandas(train_df),
            "validation": HFDataset.from_pandas(val_df)
        })

        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

        tokenized = dataset.map(
            lambda b: preprocess(b, tokenizer, MAX_INPUT_LEN, MAX_LABEL_LEN),
            batched=True,
            remove_columns=dataset["train"].column_names
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=Seq2SeqTrainingArguments(
                output_dir=f"./results/t5_fold_{fold}",
                num_train_epochs=3,
                per_device_train_batch_size=4,
                predict_with_generate=True,
                report_to="none"
            ),
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            tokenizer=tokenizer,
            compute_metrics=lambda e: compute_metrics_t5(e, tokenizer)
        )

        trainer.train()
        print(trainer.evaluate())

# =====================================
# MAIN — RUNS ALL THREE SEQUENTIALLY
# =====================================

def main():
    df = load_data()

    train_decoder(df)
    train_encoder(df)
    train_flan(df)


if __name__ == "__main__":
    main()
