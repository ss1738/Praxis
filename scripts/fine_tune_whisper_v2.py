"""
Whisper LoRA fine-tune v2 - fixes hallucination from v1.
Freezes encoder, lower LoRA rank, generation constraints at inference.
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jiwer
import numpy as np
import torch
from datasets import Audio, Dataset
from loguru import logger
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

# env vars for mps
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# config
MODEL_ID = "openai/whisper-small"
OUTPUT_DIR = "data/models/whisper-small-british-medical-v2"
DATA_DIR = Path("data/processed/primock57")

# smaller LoRA than v1 to reduce overfitting
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.2
TARGET_MODULES = ["q_proj", "v_proj"]

EPOCHS = 10
BATCH_SIZE = 2
GRAD_ACCUM = 4  # effective batch = 8
LEARNING_RATE = 1e-4  # safe because encoder is frozen
WARMUP_RATIO = 0.1
MAX_AUDIO_SECS = 20.0

WEIGHT_DECAY = 0.01

# anti-hallucination settings (applied at inference, not training)
GEN_MAX_NEW_TOKENS = 225
GEN_NO_REPEAT_NGRAM = 3
GEN_REPETITION_PENALTY = 1.3
GEN_LENGTH_PENALTY = 1.0

PATIENCE = 3

SPEED_PERTURB_RANGE = (0.9, 1.1)

# device
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# speed perturbation
def speed_perturb(audio: np.ndarray, sr: int, low: float = 0.9, high: float = 1.1) -> np.ndarray:
    """Randomly speed up or slow down for augmentation."""
    factor = random.uniform(low, high)
    if abs(factor - 1.0) < 0.01:
        return audio
    # simple resampling trick
    indices = np.round(np.arange(0, len(audio), factor)).astype(int)
    indices = indices[indices < len(audio)]
    return audio[indices]

# data collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

# dataset preparation
def load_and_prepare_dataset(
    split_path: Path,
    processor: WhisperProcessor,
    max_audio_secs: float,
    augment: bool = False,
) -> Dataset:
    with open(split_path) as f:
        records = json.load(f)

    logger.info(f"Loading {split_path.name}: {len(records)} samples (augment={augment})")

    ds = Dataset.from_dict({
        "audio": [r["audio_path"] for r in records],
        "text": [r["text"] for r in records],
    })
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))

    def process_fn(batch):
        audio = batch["audio"]
        max_samples = int(max_audio_secs * 16_000)
        samples = np.array(audio["array"][:max_samples], dtype=np.float32)

        # speed augmentation
        if augment:
            samples = speed_perturb(samples, 16_000, *SPEED_PERTURB_RANGE)

        inputs = processor.feature_extractor(
            samples, sampling_rate=16_000, return_tensors="np"
        )

        labels = processor.tokenizer(batch["text"]).input_ids
        labels = labels[:448]

        return {
            "input_features": inputs.input_features[0],
            "labels": labels,
        }

    ds = ds.map(process_fn, remove_columns=ds.column_names)
    return ds

# metrics
def make_compute_metrics(processor: WhisperProcessor):
    def _normalise(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^ a-z0-9']", " ", text)
        return " ".join(text.split())

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        pred_norm = [_normalise(p) for p in pred_str]
        label_norm = [_normalise(l) for l in label_str]

        pairs = [(r, h) for r, h in zip(label_norm, pred_norm) if r.strip()]
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
    logger.info(f"LoRA: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}, targets={TARGET_MODULES}")
    logger.info(f"Training: {EPOCHS} epochs, batch={BATCH_SIZE}×{GRAD_ACCUM}, lr={LEARNING_RATE}")
    logger.info(f"Anti-hallucination: freeze_encoder=True, weight_decay={WEIGHT_DECAY}")
    logger.info(f"Generation: no_repeat_ngram={GEN_NO_REPEAT_NGRAM}, rep_penalty={GEN_REPETITION_PENALTY}")

    # load processor and model
    logger.info("Loading Whisper processor and model...")
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

    # Force English transcription via generation_config
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="en", task="transcribe"
    )
    model.generation_config.suppress_tokens = []

    # generation constraints go in the benchmark script, not here
    # (they mess up predict_with_generate and trigger early stopping)

    # freeze encoder - only train decoder LoRA
    logger.info("Freezing encoder...")
    model.freeze_encoder()
    enc_params = sum(p.numel() for p in model.model.encoder.parameters())
    logger.info(f"Encoder frozen: {enc_params/1e6:.1f}M params locked")

    # attach lora adapters (decoder only)
    logger.info("Attaching LoRA adapters to decoder...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # prepare datasets
    logger.info("Preparing datasets...")
    train_ds = load_and_prepare_dataset(
        DATA_DIR / "train.json", processor, MAX_AUDIO_SECS, augment=True,
    )
    val_ds = load_and_prepare_dataset(
        DATA_DIR / "val.json", processor, MAX_AUDIO_SECS, augment=False,
    )
    logger.info(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=2,
        predict_with_generate=True,
        generation_max_length=GEN_MAX_NEW_TOKENS,
        fp16=False,
        bf16=False,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=3,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    # trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(processor),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
        processing_class=processor.feature_extractor,
    )

    # train
    logger.info("=" * 60)
    logger.info("Starting Whisper v2 LoRA fine-tuning (anti-hallucination)")
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
    logger.info("Running final evaluation on validation set...")
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
