## PRAXIS ASR — combined summary across all evaluated decoders on PriMock57 test (n=14)

| Configuration | Mean WER | WER 95% CI | Mean CE-WER | CE-WER 95% CI | Mean RTF |
|---|---|---|---|---|---|
| AUSCULT greedy CTC | 49.91% | [46.40%, 53.17%] | 88.64% | [72.16%, 100.00%] | 0.0014 |
| AUSCULT beam w=10 (no LM) | 49.74% | [46.24%, 52.82%] | 88.64% | [72.16%, 100.00%] | 0.0036 |
| AUSCULT beam w=10 + clinical kenlm 4-gram | 45.57% | [41.82%, 49.04%] | 86.08% | [69.95%, 97.44%] | 0.0033 |
| **openai/whisper-large-v3** (fp16, 30s chunks) | **27.69%** | [23.08%, 33.10%] | **21.23%** | [12.45%, 30.40%] | 0.0144 |

### Significance (paired Wilcoxon vs the no-LM beam baseline)

| Comparison | WER Δ (pp) | p-value |
|---|---|---|
| beam w=10 + kenlm 4-gram vs beam w=10 (no LM) | −4.17 | **0.000122** (highly significant) |

### MTER (medical-suffix exact recovery)

| Suffix | Ref hits | AUSCULT greedy | AUSCULT + kenlm | Whisper-large-v3 |
|---|---|---|---|---|
| -itis | 6 | 0% | 0% | **100%** (6/6) |
| -osis | 2 | 0% | 0% | **100%** (2/2) |
| -profen | 2 | 0% | 0% | **100%** (2/2) |
| **total** | 10 | **0/10** | **0/10** | **10/10** |
