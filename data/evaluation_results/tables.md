## PRAXIS Benchmark Results

*Auto-generated 2026-04-27 13:51; TrustScore column re-synced to the 2026-05-24 rerun. Regenerate with `make eval-tables`.*

### ASR Word Error Rate (WER) — PriMock57 Test Set

| Model | Mean WER | Median WER | 95% CI | RTF |
|-------|----------|------------|--------|-----|
| praxis-conformer | 50.12% | 50.17% | [46.70, 53.47] | 0.044 |

### SOAP Note Quality

| Model | N | Completeness | SNOMED Codes | Med. Recall | Safety Net | TrustScore |
|-------|---|-------------|-------------|-------------|------------|------------|
| med42 | 8 | 100% | 3.4 | 100% | 100% | 69.4 |
| meditron:7b | 8 | 31% | 0.8 | 100% | 12% | 49.4 |
| mistral | 8 | 100% | 5.5 | 100% | 100% | 65.6 |
| praxis-soap-v1 | 8 | 100% | 2.8 | 100% | 100% | 70.6 |
