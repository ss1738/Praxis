"""Resume training - steps 1-3 done, pick up from Whisper v3."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jiwer
import librosa
import numpy as np
import soundfile as sf
import torch
from datasets import Audio, Dataset
from loguru import logger

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

COMBINED_DIR = Path("data/processed/combined_v3")
PRIMOCK_DIR = Path("data/processed/primock57")
WHISPER_V3_DIR = "data/models/whisper-british-medical-v3"
MEDASR_V2_DIR = "data/models/medasr-british-v2"

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def _normalise(text: str) -> str:
    text = text.lower()
    text = text.replace("</s>", "")
    text = re.sub(r"[^ a-z0-9']", " ", text)
    return " ".join(text.split())

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

@dataclass
class DataCollatorCTCWithPadding:
    processor: Any
    padding: bool = True

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_feats = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_feats, padding=self.padding, return_tensors="pt"
        )

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, padding=self.padding, return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels
        return batch

def prepare_whisper_dataset(split_path: Path, processor, max_audio_secs: float = 20.0) -> Dataset:
    with open(split_path) as f:
        records = json.load(f)

    records = [r for r in records if Path(r["audio_path"]).exists()]
    logger.info(f"  Loading {split_path.name}: {len(records)} valid samples")

    ds = Dataset.from_dict({
        "audio": [r["audio_path"] for r in records],
        "text": [r["text"] for r in records],
    })
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))

    def process_fn(batch):
        audio = batch["audio"]
        max_samples = int(max_audio_secs * 16_000)
        samples = np.array(audio["array"][:max_samples], dtype=np.float32)

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

def make_whisper_compute_metrics(processor):
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
        return {"wer": jiwer.wer(list(refs), list(hyps))}

    return compute_metrics

def prepare_medasr_dataset(split_path: Path, processor, max_audio_secs: float = 30.0) -> Dataset:
    with open(split_path) as f:
        records = json.load(f)

    records = [r for r in records if Path(r["audio_path"]).exists()]
    logger.info(f"  Loading {split_path.name}: {len(records)} valid samples")

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

def make_medasr_compute_metrics(processor):
    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred_strs = []
        for seq in pred_ids:
            collapsed = []
            prev = -1
            for idx in seq:
                if idx != prev and idx != 0:
                    collapsed.append(int(idx))
                prev = idx
            pred_strs.append(processor.tokenizer.decode(collapsed))

        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_strs = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        pred_norm = [_normalise(p) for p in pred_strs]
        label_norm = [_normalise(l) for l in label_strs]

        pairs = [(r, h) for r, h in zip(label_norm, pred_norm) if r.strip() and h.strip()]
        if not pairs:
            return {"wer": 1.0}
        refs, hyps = zip(*pairs)
        return {"wer": jiwer.wer(list(refs), list(hyps))}

    return compute_metrics

def freeze_encoder_layers(model, freeze_ratio: float = 0.7):
    all_params = list(model.named_parameters())
    encoder_params = [(n, p) for n, p in all_params if "encoder" in n]
    n_freeze = int(len(encoder_params) * freeze_ratio)
    for i, (name, param) in enumerate(encoder_params):
        if i < n_freeze:
            param.requires_grad = False
    frozen = sum(1 for _, p in all_params if not p.requires_grad)
    total = len(all_params)
    logger.info(f"Froze {frozen}/{total} parameters ({frozen/total:.0%})")

# step 4: whisper v3
def train_whisper_v3():
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import (
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    device = get_device()
    logger.info(f"STEP 4: Training Whisper v3 (device={device})")

    MODEL_ID = "openai/whisper-small"
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="en", task="transcribe"
    )
    model.generation_config.suppress_tokens = []

    model.freeze_encoder()
    logger.info("Encoder frozen")

    lora_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.2,
        target_modules=["q_proj", "v_proj"],
        bias="none", task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = prepare_whisper_dataset(COMBINED_DIR / "train.json", processor)
    val_ds = prepare_whisper_dataset(COMBINED_DIR / "val.json", processor)
    logger.info(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=WHISPER_V3_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        predict_with_generate=True,
        generation_max_length=225,
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

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=make_whisper_compute_metrics(processor),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        processing_class=processor.feature_extractor,
    )

    logger.info("Starting Whisper v3 training...")
    train_result = trainer.train()

    trainer.save_model(WHISPER_V3_DIR)
    processor.save_pretrained(WHISPER_V3_DIR)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    logger.info(f"Whisper v3 done. Eval WER: {eval_metrics.get('eval_wer', 'N/A')}")
    logger.info(f"Saved to {WHISPER_V3_DIR}")

    # Free memory
    del model, trainer
    import gc
    gc.collect()

# step 5: medasr v2
def train_medasr_v2():
    from transformers import (
        AutoModelForCTC,
        AutoProcessor,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    device = get_device()
    logger.info(f"STEP 5: Training MedASR v2 (device={device})")

    MODEL_ID = "google/medasr"
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForCTC.from_pretrained(MODEL_ID)

    freeze_encoder_layers(model, freeze_ratio=0.7)

    train_ds = prepare_medasr_dataset(COMBINED_DIR / "train.json", processor)
    val_ds = prepare_medasr_dataset(COMBINED_DIR / "val.json", processor)
    logger.info(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    data_collator = DataCollatorCTCWithPadding(processor=processor)

    training_args = TrainingArguments(
        output_dir=MEDASR_V2_DIR,
        num_train_epochs=10,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        warmup_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        fp16=device == "cuda",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=3,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=make_medasr_compute_metrics(processor),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
    )

    logger.info("Starting MedASR v2 training...")
    train_result = trainer.train()

    trainer.save_model(MEDASR_V2_DIR)
    processor.save_pretrained(MEDASR_V2_DIR)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    logger.info(f"MedASR v2 done. Eval WER: {eval_metrics.get('eval_wer', 'N/A')}")
    logger.info(f"Saved to {MEDASR_V2_DIR}")

    del model, trainer
    import gc
    gc.collect()

# step 6: benchmark
def benchmark_all():
    from rich.console import Console
    from rich.table import Table

    console = Console(width=120)
    logger.info("STEP 6: Benchmarking all models")

    with open(PRIMOCK_DIR / "test.json") as f:
        test_records = json.load(f)

    console.print(f"\nTest samples: {len(test_records)}")

    model_configs = []
    model_configs.append(("Base Whisper Small", "whisper", "openai/whisper-small", {}))

    v2_path = Path("data/models/whisper-small-british-medical-v2")
    if v2_path.exists() and (v2_path / "adapter_config.json").exists():
        model_configs.append(("Whisper v2 (LoRA)", "whisper_lora", str(v2_path), {
            "no_repeat_ngram_size": 3, "repetition_penalty": 1.3, "max_new_tokens": 225,
        }))

    v3_path = Path(WHISPER_V3_DIR)
    if v3_path.exists() and (v3_path / "adapter_config.json").exists():
        model_configs.append(("Whisper v3 (expanded)", "whisper_lora", str(v3_path), {
            "no_repeat_ngram_size": 3, "repetition_penalty": 1.3, "max_new_tokens": 225,
        }))

    model_configs.append(("Base MedASR", "medasr", "google/medasr", {}))

    v1_medasr = Path("data/models/medasr-british-medical")
    if v1_medasr.exists() and (v1_medasr / "model.safetensors").exists():
        model_configs.append(("MedASR v1", "medasr", str(v1_medasr), {}))

    v2_medasr = Path(MEDASR_V2_DIR)
    if v2_medasr.exists():
        has_model = (v2_medasr / "model.safetensors").exists() or (v2_medasr / "pytorch_model.bin").exists()
        if has_model:
            model_configs.append(("MedASR v2 (expanded)", "medasr", str(v2_medasr), {}))

    all_results = {}

    for name, mtype, path, gen_kwargs in model_configs:
        console.print(f"\nBenchmarking [bold]{name}[/bold]...")
        try:
            if mtype == "whisper":
                from transformers import WhisperForConditionalGeneration, WhisperProcessor
                processor = WhisperProcessor.from_pretrained(path)
                model = WhisperForConditionalGeneration.from_pretrained(path)
                model.eval()
            elif mtype == "whisper_lora":
                from transformers import WhisperForConditionalGeneration, WhisperProcessor
                from peft import PeftModel
                with open(Path(path) / "adapter_config.json") as f:
                    ac = json.load(f)
                base_id = ac.get("base_model_name_or_path", "openai/whisper-small")
                processor = WhisperProcessor.from_pretrained(base_id)
                base_model = WhisperForConditionalGeneration.from_pretrained(base_id)
                model = PeftModel.from_pretrained(base_model, path)
                model = model.merge_and_unload()
                model.eval()
            elif mtype == "medasr":
                from transformers import AutoModelForCTC, AutoProcessor
                processor = AutoProcessor.from_pretrained(path)
                model = AutoModelForCTC.from_pretrained(path)
                model.eval()
        except Exception as e:
            console.print(f"  [red]Failed to load: {e}[/red]")
            continue

        results = []
        for i, rec in enumerate(test_records):
            audio_path = rec["audio_path"]
            full_ref = rec["text"]
            full_dur = librosa.get_duration(path=audio_path)
            segment_dur = min(30.0, full_dur)
            audio, sr = librosa.load(audio_path, sr=16000, duration=30.0)

            words_per_sec = max(len(full_ref.split()) / full_dur, 1)
            approx_words = int(words_per_sec * segment_dur)
            ref_text = " ".join(full_ref.split()[:approx_words])
            ref_norm = _normalise(ref_text)

            try:
                t0 = time.perf_counter()
                if "medasr" in mtype:
                    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
                    with torch.no_grad():
                        logits = model(**inputs).logits
                    pred_ids = torch.argmax(logits, dim=-1)[0]
                    collapsed = []
                    prev = -1
                    for idx in pred_ids.tolist():
                        if idx != prev and idx != 0:
                            collapsed.append(idx)
                        prev = idx
                    hyp_text = processor.tokenizer.decode(collapsed)
                else:
                    inputs = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
                    with torch.no_grad():
                        gen_ids = model.generate(
                            **inputs, language="en", task="transcribe", **gen_kwargs,
                        )
                    hyp_text = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]

                elapsed = time.perf_counter() - t0
                hyp_norm = _normalise(hyp_text.strip())
                wer = jiwer.wer(ref_norm, hyp_norm) if ref_norm and hyp_norm else 1.0
                cer = jiwer.cer(ref_norm, hyp_norm) if ref_norm and hyp_norm else 1.0
            except Exception as e:
                logger.warning(f"{name} failed on sample {i}: {e}")
                hyp_text, elapsed, wer, cer = "", 0.0, 1.0, 1.0

            results.append({"file": Path(audio_path).name, "wer": wer, "cer": cer, "time": elapsed})

        all_results[name] = results

        avg_wer = sum(r["wer"] for r in results) / len(results)
        avg_cer = sum(r["cer"] for r in results) / len(results)
        console.print(f"  Avg WER: {avg_wer:.1%}, Avg CER: {avg_cer:.1%}")

        del model, processor
        import gc
        gc.collect()

    # Summary
    console.print("\n")
    table = Table(title="BENCHMARK v3: All Models", show_header=True, show_lines=True)
    table.add_column("Model", style="bold", min_width=25)
    table.add_column("Avg WER", justify="right")
    table.add_column("Avg CER", justify="right")
    table.add_column("Avg Time", justify="right")

    for name, results in all_results.items():
        avg_wer = sum(r["wer"] for r in results) / len(results)
        avg_cer = sum(r["cer"] for r in results) / len(results)
        avg_time = sum(r["time"] for r in results) / len(results)
        table.add_row(name, f"{avg_wer:.1%}", f"{avg_cer:.1%}", f"{avg_time:.2f}s")

    console.print(table)

    out_path = Path("data/benchmark_v3.json")
    out_path.write_text(json.dumps(all_results, indent=2, default=str))
    console.print(f"\nResults saved to {out_path}")
    logger.info("Benchmark complete")

if __name__ == "__main__":
    t_start = time.time()

    # Clean up incomplete checkpoint from aborted run
    import shutil
    v3_dir = Path(WHISPER_V3_DIR)
    if v3_dir.exists():
        shutil.rmtree(v3_dir)
        logger.info("Cleaned up incomplete Whisper v3 directory")

    try:
        train_whisper_v3()
    except Exception as e:
        logger.error(f"Whisper v3 failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        train_medasr_v2()
    except Exception as e:
        logger.error(f"MedASR v2 failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        benchmark_all()
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

    logger.info(f"Total time: {(time.time()-t_start)/60:.1f} min")
