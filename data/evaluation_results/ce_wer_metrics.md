## CE-WER aggregates

| Decoder | n | Mean WER | Mean CE-WER | CE-WER 95% CI | Clinical tokens / sample |
|---|---|---|---|---|---|
| `greedy` | 14 | 56.22% | 88.64% | [72.16%, 100.00%] | 5.1 |
| `beam_nobias` | 14 | 49.76% | 88.64% | [72.16%, 100.00%] | 5.1 |

## MTER per suffix (greedy)

| Suffix | Ref hits | Exact | Suffix-only | Omission | Recovery |
|---|---|---|---|---|---|
| -itis | 6 | 0 | 0 | 6 | 0.0% |
| -osis | 2 | 0 | 0 | 2 | 0.0% |
| -profen | 2 | 0 | 0 | 2 | 0.0% |

## MTER per suffix (beam_nobias)

| Suffix | Ref hits | Exact | Suffix-only | Omission | Recovery |
|---|---|---|---|---|---|
| -itis | 6 | 0 | 0 | 6 | 0.0% |
| -osis | 2 | 0 | 0 | 2 | 0.0% |
| -profen | 2 | 0 | 0 | 2 | 0.0% |
