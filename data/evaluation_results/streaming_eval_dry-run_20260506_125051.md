# Streaming-veto vs Baseline vs Constitutional
_Run timestamp_: `2026-05-06T12:50:51+00:00`
_Bootstrap CI_: 1000 iterations, seed 42

| Strategy | Mean TrustScore (95% CI) | Mean Hallucination Rate (95% CI) | Mean Time (s) | Perfect Grounding |
|---|---|---|---|---|
| baseline | 72.1 [66.2, 79.6] | 0.156 [0.100, 0.207] | 10.8 | 0/8 |
| constitutional | 78.8 [75.0, 83.8] | 0.099 [0.069, 0.126] | 16.7 | 0/8 |
| streaming | 78.9 [75.1, 83.6] | 0.000 [0.000, 0.000] | 25.1 | 8/8 |

Note: Streaming's `Perfect Grounding` is by-construction; every shipped sentence has passed the per-sentence NLI gate.