## Macro Kendall τ

| Comparison | τ |
|---|---|
| Mistral judge vs TrustScore | -0.312 |
| Llama judge vs TrustScore | -0.042 |
| Rule-based vs TrustScore | 0.333 |
| Mistral vs Llama | 0.250 |

## Verdict Cohen κ (3-state)

| Comparison | κ | Observed | Expected |
|---|---|---|---|
| mistral_vs_trustscore | -0.063 | 0.375 | 0.412 |
| llama_vs_trustscore | -0.031 | 0.375 | 0.394 |
| rule_vs_trustscore | -0.063 | 0.375 | 0.412 |
| mistral_vs_llama | 0.492 | 0.969 | 0.939 |

## Per-criterion saturation (Llama judge)

| Criterion | Mean | Std | Max−Min |
|---|---|---|---|
| structure | 0.917 | 0.066 | 0.131 |
| content | 0.830 | 0.041 | 0.081 |
| snomed | 0.753 | 0.062 | 0.138 |
| safety_netting | 0.839 | 0.092 | 0.212 |
| uk_english | 0.969 | 0.062 | 0.125 |
| budget | 0.891 | 0.065 | 0.138 |

## TrustScore per-check stratification

| Check | Mean | Std | Max−Min |
|---|---|---|---|
| transcript_fidelity | 0.812 | 0.072 | 0.125 |
| snomed_validation | 0.719 | 0.329 | 0.750 |
| clinical_safety | 0.469 | 0.237 | 0.500 |
| nice_guidelines | 0.062 | 0.072 | 0.125 |
| completeness | 0.781 | 0.438 | 0.875 |
| drug_interactions | 1.000 | 0.000 | 0.000 |
