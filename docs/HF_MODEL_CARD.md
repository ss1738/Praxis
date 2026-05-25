---
language:
  - en
license: other
license_name: msc-dissertation-rights-reserved
license_link: LICENSE
library_name: pytorch
pipeline_tag: automatic-speech-recognition
tags:
  - automatic-speech-recognition
  - asr
  - conformer
  - ctc
  - medical
  - clinical
  - nhs
  - gp
  - primary-care
  - english
  - en-GB
  - primock57
  - praxis
  - praxis-auscult
  - auscult
datasets:
  - primock57
metrics:
  - wer
model-index:
  - name: PRAXIS-AUSCULT
    results:
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: primock57
          name: PriMock57 (test split, n=14, locked)
        metrics:
          - type: wer
            value: 49.91
            name: Word Error Rate (greedy CTC, no LM, no biasing, no audio preprocessing)
          - type: wer
            value: 49.74
            name: Word Error Rate (beam w=10, no LM, no biasing)
          - type: wer
            value: 45.57
            name: Word Error Rate (beam w=10 + clinical kenlm 4-gram rescoring)
          - type: cer
            value: 88.64
            name: Clinical Entity WER (greedy / beam, no LM)
---

# PRAXIS-AUSCULT: Conformer-CTC ASR for British GP Audio

A **117M-parameter (live inference)** Conformer-CTC speech recogniser trained from scratch on **British General Practice consultation audio** (PriMock57). Built as the ASR component of **PRAXIS**, an MSc dissertation system at the University of Leicester for AI-assisted clinical documentation in NHS general practice.

**AUSCULT** is named after Latin *auscultare*, *"to listen carefully"*, the root of *auscultation*, the foundational act of clinical listening with a stethoscope. The model learns to listen to GP consultations the way a clinician learns to listen to a chest.

> ⚠️ **Filename note.** The `_120M` suffix reflects the checkpoint state-dict size (119.6M params); **117M of those are loaded for inference** (≈2.5M training-only output-head params are stripped at load).

## Model summary

| | |
|---|---|
| Architecture | Conformer-CTC (encoder-only, CTC head) |
| Parameters (inference / checkpoint) | **117.06M / 119.63M** |
| Layers | 18 Conformer blocks |
| `d_model` / heads / FFN | 512 / 8 / 2048 |
| Conv kernel | 31 |
| Vocabulary | 3,001 (SentencePiece, see `praxis_tokenizer_v2.model`) |
| Real-time factor (RTF) | 0.036–0.044 (Mac CPU, verified live) |
| Framework | Raw PyTorch (no `transformers`) |

## Training

| | |
|---|---|
| Training data | PriMock57 (57 simulated British GP consultations), chunked into 4,824 segments |
| Training compute | vast.ai (NVIDIA GPUs); see `vast_train.py` for the exact training script |
| Loss | CTC |
| Tokeniser | SentencePiece, 3,001 BPE-style tokens |

## Headline results: PriMock57 test split (n = 14, locked)

All numbers below were re-measured 2026-05-24 on an RTX 5090 against the canonical checkpoint `best_primock57.pt` (sha256 `e47f9694f5afad28…`). Sources: `data/evaluation_results/asr_combined_summary.tex`, `kenlm_rescoring_metrics.json`, `whisper_baseline_metrics.json`.

### WER + Clinical Entity WER (CE-WER) + Medical-suffix exact recovery

| Configuration | Params | WER (95% bootstrap CI) | CE-WER (95% CI) | Medical-suffix recovery | Mean RTF |
|---|---|---|---|---|---|
| PRAXIS-AUSCULT, greedy CTC | 117M | **49.91%** [46.40, 53.17] | 88.64% [72.16, 100.00] | **0 / 10** | 0.0014 |
| PRAXIS-AUSCULT, beam w=10 (no LM, no biasing) | 117M | **49.74%** [46.24, 52.82] | 88.64% [72.16, 100.00] | **0 / 10** | 0.0036 |
| PRAXIS-AUSCULT, beam w=10 + clinical kenlm 4-gram | 117M | **45.57%** [41.82, 49.04] | 86.08% [69.95, 97.44] | **0 / 10** | 0.0033 |
| `openai/whisper-large-v3` (fp16, 30 s chunks, identical test split) | 1.55B | **27.69%** [23.08, 33.10] | 21.23% [12.45, 30.40] | **10 / 10** | 0.0144 |

**Reconciliation with prior reported number.** The "50.12% WER" value previously cited corresponds to a decoder configuration in the beam-search family on the same checkpoint and same locked test split. Today's re-measurement matches that value to within 0.36 pp (beam w=10 + kenlm: 49.76%; without LM: 49.74%). The greedy baseline (49.91%) is also within 0.21 pp. The historical 56.72% second number is attributable to the audio preprocessing pipeline (see "Preprocessing variance" below), not to decoder choice or checkpoint identity.

**Inferential statistics** (paired Wilcoxon, n = 14):

- AUSCULT beam w=10 + clinical kenlm 4-gram vs AUSCULT beam w=10 (no LM): Δ WER = **−4.17 pp**, *p* = **1.2 × 10⁻⁴** (highly significant).
- Δ CE-WER = −2.56 pp, *p* = 1.000 (no detected effect on clinical-entity recall).

**Honest framing.** This Conformer is trained from scratch on 4,824 chunked segments of British GP audio: roughly four orders of magnitude less data than the foundation-model class. The 49.91% greedy WER and 88.64% CE-WER are reported as deliberate negative results: the floor against which the dissertation's downstream verification, hallucination, and source-linking contributions are evaluated. The previously-quoted Whisper-large-v3 number (16.76%) was a citation; the 27.69% on this row is the value measured by this work on the identical locked test split.

**Verified live (2026-05-04):** RTF 0.036 on macOS CPU with a 253-second sample consultation. Model loaded in 0.47s. Live transcription is intelligible but contains the word-level errors expected at the reported WER (e.g. "diziness" for "dizziness").

### What the model is good for

- **On-device, sub-real-time inference**: full pipeline runs locally on Apple Silicon at RTF well under 0.1, allowing co-residency with larger downstream models on commodity hardware.
- **Domain-specific phrasing**: common British GP turns of phrase, EMIS-style consultation patterns, NHS abbreviations (CKS, BNF, NICE).
- **Negative-result documentation for in-domain compact ASR**: this artifact establishes that a 117M Conformer-CTC trained on 4,824 chunked PriMock57 segments does not match foundation-model class accuracy on the same test split. The downstream PRAXIS verification stack (TrustScore, Constitutional self-correction, source-linking) compensates by being ASR-agnostic: see "Upstream acoustic emission floor" below.

### Medical-term recall: measured (not reproducible at the previously-cited level)

On a 40-chunk clinical-reference subset of PriMock57 train (≥ 2 lexicon-hits per chunk, 138 total lexicon-hit reference tokens), the SnomedBiaser (boost = 8.0, 185-term lexicon, beam w=10) yields:

- Mean WER greedy: 0.6799 / mean WER biased: 0.6836 (Δ +0.4 pp: biasing slightly worse on overall WER)
- mc_recall greedy: 0.000 / mc_recall biased: 0.000
- **Relative MC-WER reduction: 0.0%** (Wilcoxon paired *p* = 0.34, no significant difference)

The previously-cited "31% relative reduction in medical-term WER" claim is **not reproducible** under this measurement. Source: `data/evaluation_results/snomed_biasing_mc_wer_latest.json`.

### Preprocessing variance

The historical 50.12% vs 56.72% WER discrepancy in two earlier evaluation runs on the same checkpoint is attributable to whether the audio preprocessing pipeline (`noisereduce` + `librosa.effects.trim` + peak normalisation) was applied at inference. Same n = 14 test split:

| Configuration | Mean WER |
|---|---|
| greedy, *with* full preprocessing | 56.22% |
| greedy, *without* preprocessing (direct mel path) | 49.91% |
| **Preprocessing-attributable penalty** | **+6.31 pp** |

The decoder-attributable variance across beam widths w ∈ {10, 50, 100, 200, 500} is < 0.2 pp on the no-preprocessing path; preprocessing therefore dominates decoder choice as a source of WER variance by approximately 37×. The specific preprocessing operation responsible has not been isolated in this audit; users running comparison benchmarks should report the preprocessing configuration alongside the WER number. Sources: `data/evaluation_results/wer_decoding_sweep_final.json`, `asr_beam_sensitivity_sweep.json`.

### Upstream acoustic emission floor (ASR-agnostic downstream stack)

On the identical n = 14 test split, three downstream interventions failed to recover medical-suffix tokens (–itis, –osis, –profen: 10 reference tokens total):

- Beam width sweep 10 → 500: **0 / 10** recovered (suffix recovery is invariant to beam width).
- Clinical kenlm 4-gram rescoring (ARPA built from 5,557 sentences / 270,042 tokens of PriMock57 train + 102 BNF rows + 29 NICE CKS conditions + 185-term SnomedLexicon): **0 / 10** recovered.

Substituting the upstream ASR with `openai/whisper-large-v3`: no other PRAXIS pipeline change: yields:

- WER: 27.69% (Δ −22.05 pp vs AUSCULT greedy).
- CE-WER: 21.23% (Δ −67.41 pp).
- **Medical-suffix exact recovery: 10 / 10.**

The PRAXIS deterministic verification stack (TrustScore, Constitutional self-correction, source-text linking, audit log JSONL) operated unmodified across the AUSCULT → Whisper substitution. The framework's clinical-safety contributions are decoupled at the interface boundary and not dependent on the upstream ASR choice. Sources: `kenlm_rescoring_metrics.json`, `whisper_baseline_metrics.json`, `asr_combined_summary.tex`.

## Intended use

- Research and educational use within PRAXIS (MSc dissertation, University of Leicester).
- Studying domain-specific ASR with limited training data.
- Studying contextual-biasing methods (the SNOMED hotword boost is a contribution in its own right).

### Out of scope

- **Not a clinical device.** Output must be reviewed by a qualified clinician before being entered into a patient record.
- Not validated outside British primary-care GP audio. Will likely degrade on hospital, secondary-care, paediatric, ICU, or non-UK English audio.
- Not validated for code-switched, multilingual, or strongly accented non-UK English.

## Files in this repo

| File | Purpose |
|---|---|
| `best_model_v2_120M.pt` | Generic checkpoint, val_loss 0.679 on the multi-corpus validation set. |
| `final_model_v2_120M.pt` | Final-epoch generic checkpoint. |
| `praxis_tokenizer_v2.model` | SentencePiece tokeniser, vocab 3,001. Required for inference. |
| `vast_train.py` | The exact training script used on vast.ai. Provided for reproducibility. |

> **Which checkpoint to use.** The PRAXIS reference engine prefers a PriMock57-fine-tuned checkpoint (`best_primock57.pt`) when it is present locally; otherwise it falls back to `best_model_v2_120M.pt`. The PriMock-fine-tuned variant is the one used to obtain the dissertation WER number (50.12% on PriMock57 test) and is recommended for British GP audio. The `_120M` suffix reflects the checkpoint state-dict size (119.6M params); the live inference graph is **117M params** (~2.5M training-only output-head params are stripped at load).

## How to use

This model is **not packaged as a `transformers` model**. It is loaded with raw PyTorch.

The full inference engine (`ConformerEngine`, SNOMED biaser, audio preprocessor) lives in the **PRAXIS reference implementation**, which is **private until dissertation submission**. Contact the author for research access.

### Inspecting the checkpoint without the reference repo

This works standalone:

```python
import torch
from huggingface_hub import hf_hub_download

ckpt_path = hf_hub_download(repo_id="Satyawan1/praxis-auscult", filename="best_model_v2_120M.pt")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

print(ckpt["config"])
# {'vocab_size': 3001, 'd_model': 512, 'n_heads': 8, 'ff_dim': 2048, 'n_layers': 18, 'conv_kernel': 31}
print(ckpt["val_loss"])  # 0.679
state_dict = ckpt["model_state"]  # PyTorch state dict, ready to load into a matching Conformer-CTC architecture
```

### Running inference

Requires the PRAXIS engine code. Once available:

```python
from huggingface_hub import snapshot_download
from speech_to_text.conformer_engine import ConformerEngine   # from PRAXIS repo

local_dir = snapshot_download(repo_id="Satyawan1/praxis-auscult")
engine = ConformerEngine(model_dir=local_dir, bias=False)
text = engine.transcribe("/path/to/consultation.wav")

# Contextual SNOMED biasing exists in the engine; on the locked PriMock57 test
# split the biased decoder yields a 0.0% relative reduction in medical-term WER
# (see "Medical-term recall: measured" above). Do not assume positive effect.
engine_biased = ConformerEngine(model_dir=local_dir, bias=True)
text = engine_biased.transcribe("/path/to/consultation.wav")
```

## Compliance and regulatory status

- **Research prototype only.** This artifact is NOT a CE-marked or FDA-cleared medical device.
- **No HIPAA / GDPR audit has been performed on this artifact.** Clinical deployment requires a separate Data Protection Impact Assessment and Information Governance review under the relevant jurisdiction.
- Output must be reviewed by a qualified clinician before being entered into any patient record.
- Distribution licence: CC-BY-NC-4.0; academic and non-commercial research use with attribution. Contact the author for commercial licensing.

## Limitations and risks

1. **WER is high in absolute terms** (49.91% greedy / 49.74% beam / 45.57% with clinical LM rescoring; CE-WER 88.64%; 0 / 10 medical-suffix exact recovery). Use only as part of a downstream pipeline that includes clinician-in-the-loop review.
2. Trained on **simulated** GP consultations (PriMock57), not real clinical recordings. Real-world performance is not yet established.
3. Speaker-level diarisation is handled by a separate model (PRAXIS-TIMBRE, `Satyawan1/praxis-timbre`); see that repository for its own measured DER profile.
4. The SNOMED hotword lexicon is curated for UK primary care; the biasing decode-time mechanism in this checkpoint did not yield a measurable medical-term-recall improvement on the locked test split (Δ MC-WER 0.0%, Wilcoxon *p* = 0.34).
5. No safety guard rails are baked into the ASR itself. Hallucination detection lives in the downstream PRAXIS TrustScore pipeline (NLI entailment gate, Asgari-style harness, MHI), not here.
6. **Upstream acoustic emission floor.** Medical-suffix sub-words (`-itis`, `-osis`, `-profen`) are completely omitted by this acoustic model on the test split; downstream rescoring cannot recover them. For deployment scenarios where medical-entity recall is the priority, substitute the upstream ASR (e.g. `openai/whisper-large-v3`); PRAXIS's downstream verification stack is decoupled and works unchanged.

## Citation

If you use this model in academic work, please cite the dissertation (in preparation):

```bibtex
@mastersthesis{singh2026praxis,
  author       = {Singh, Satyawan},
  title        = {{PRAXIS}: AI-Powered Clinical Documentation for {NHS} General Practice},
  school       = {University of Leicester},
  year         = {2026},
  type         = {{MSc} Dissertation},
  note         = {Programme: AI for Business Intelligence; in preparation}
}
```

## License

Released under custom MSc-dissertation rights ("All rights reserved") pending a formal open licence post-submission. Contact the author for research / non-commercial use.

## Author

Satyawan Singh, University of Leicester MSc (AI for Business Intelligence).
Academic Supervisor: University of Leicester. Industrial Supervisor: industry partner.
HF: [`Satyawan1`](https://huggingface.co/Satyawan1)
GitHub: [`ss1738`](https://github.com/ss1738)
