"""Fine-tune MedASR on PriMock57. Freeze bottom 80% of encoder, train the rest."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jiwer
import numpy as np
import torch
from datasets import Audio, Dataset
from loguru import logger
from transformers import (
    AutoModelForCTC,
    AutoProcessor,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# config
MODEL_ID = "google/medasr"
OUTPUT_DIR = "data/models/medasr-british-medical"
DATA_DIR = Path("data/processed/primock57")

FREEZE_RATIO = 0.8       # Freeze bottom 80% of encoder layers
EPOCHS = 15
BATCH_SIZE = 4
GRAD_ACCUM = 4           # Effective batch = 16
LEARNING_RATE = 5e-5
WARMUP_STEPS = 100
MAX_AUDIO_SECS = 30.0
PATIENCE = 4

# device
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# data collator
@dataclass
class DataCollatorCTCWithPadding:
    """Pad spectrograms and labels for CTC."""

    processor: AutoProcessor
    padding: bool = True

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # Pad input features (spectrograms)
        input_feats = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_feats, padding=self.padding, return_tensors="pt"
        )

        # Pad labels
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, padding=self.padding, return_tensors="pt"
        )

        # Replace pad token with -100 for CTC loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels
        return batch

# dataset preparation
def load_and_prepare_dataset(
    split_path: Path,
    processor: AutoProcessor,
    max_audio_secs: float,
) -> Dataset:

    with open(split_path) as f:
        records = json.load(f)

    logger.info(f"Loading {split_path.name}: {len(records)} samples")

    ds = Dataset.from_dict({
        "audio": [r["audio_path"] for r in records],
        "text": [r["text"] for r in records],
    })
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))

    def process_fn(batch):
        audio = batch["audio"]
        max_samples = int(max_audio_secs * 16_000)
        samples = np.array(audio["array"][:max_samples], dtype=np.float32)

        inputs = processor(samples, sampling_rate=16_000, return_tensors="np")

        labels = processor.tokenizer(batch["text"]).input_ids

        return {
            "input_features": inputs.input_features[0],
            "labels": labels,
        }

    ds = ds.map(process_fn, remove_columns=ds.column_names)
    return ds

# freeze layers
def freeze_encoder_layers(model, freeze_ratio: float = 0.8):

    # Find encoder layers
    encoder = None
    for name, module in model.named_modules():
        if "encoder" in name and hasattr(module, "layers"):
            encoder = module
            break

    if encoder is None:
        all_params = list(model.named_parameters())
        encoder_params = [n for n, _ in all_params if "encoder" in n]
        n_freeze = int(len(encoder_params) * freeze_ratio)
        for i, (name, param) in enumerate(all_params):
            if "encoder" in name and i < n_freeze:
                param.requires_grad = False
        frozen = sum(1 for _, p in all_params if not p.requires_grad)
        total = len(all_params)
        logger.info(f"Froze {frozen}/{total} parameters ({frozen/total:.0%})")
        return

    n_layers = len(encoder.layers)
    n_freeze = int(n_layers * freeze_ratio)
    for i, layer in enumerate(encoder.layers):
        if i < n_freeze:
            for param in layer.parameters():
                param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Froze {n_freeze}/{n_layers} encoder layers")
    logger.info(f"Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({trainable/total:.1%})")

# metrics
def make_compute_metrics(processor: AutoProcessor):

    def _normalise(text: str) -> str:
        text = text.lower()
        text = text.replace("</s>", "")
        text = re.sub(r"[^ a-z0-9']", " ", text)
        return " ".join(text.split())

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        # CTC decode
        pred_strs = []
        for seq in pred_ids:
            collapsed = []
            prev = -1
            for idx in seq:
                if idx != prev and idx != 0:
                    collapsed.append(int(idx))
                prev = idx
            text = processor.tokenizer.decode(collapsed)
            pred_strs.append(text)

        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_strs = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        pred_norm = [_normalise(p) for p in pred_strs]
        label_norm = [_normalise(l) for l in label_strs]

        pairs = [(r, h) for r, h in zip(label_norm, pred_norm) if r.strip() and h.strip()]
        if not pairs:
            return {"wer": 1.0}

        refs, hyps = zip(*pairs)
        wer = jiwer.wer(list(refs), list(hyps))
        return {"wer": wer}

    return compute_metrics

# main
def main():
    device = get_device()
    logger.info(f"Device: {device}")
    logger.info(f"Model: {MODEL_ID}")
    logger.info(f"Freeze ratio: {FREEZE_RATIO}")
    logger.info(f"Training: {EPOCHS} epochs, batch={BATCH_SIZE}×{GRAD_ACCUM}, lr={LEARNING_RATE}")

    # load processor and model
    logger.info("Loading MedASR processor and model...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForCTC.from_pretrained(MODEL_ID)

    # Freeze encoder layers
    freeze_encoder_layers(model, FREEZE_RATIO)

    # prepare datasets
    logger.info("Preparing datasets...")
    train_ds = load_and_prepare_dataset(DATA_DIR / "train.json", processor, MAX_AUDIO_SECS)
    val_ds = load_and_prepare_dataset(DATA_DIR / "val.json", processor, MAX_AUDIO_SECS)
    logger.info(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor)

    # training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=5,
        fp16=device == "cuda",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=3,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    # trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(processor),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
    )

    # train
    logger.info("=" * 60)
    logger.info("Starting MedASR fine-tuning on PriMock57")
    logger.info("=" * 60)
    train_result = trainer.train()

    # save
    logger.info(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # final eval
    logger.info("Running final evaluation...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    logger.info("=" * 60)
    logger.info(f"Training complete!")
    logger.info(f"Final eval WER: {eval_metrics.get('eval_wer', 'N/A')}")
    logger.info(f"Model saved to: {OUTPUT_DIR}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
