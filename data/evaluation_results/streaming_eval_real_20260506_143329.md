# Streaming-veto vs Baseline vs Constitutional
_Run timestamp_: `2026-05-06T14:33:29+00:00`
_Bootstrap CI_: 1000 iterations, seed 42

| Strategy | Mean TrustScore (95% CI) | Mean Hallucination Rate (95% CI) | Mean Time (s) | Perfect Grounding |
|---|---|---|---|---|
| baseline | 63.8 [59.4, 67.5] | 0.911 [0.842, 0.979] | 8.9 | 0/8 |
| constitutional | 70.6 [66.2, 75.0] | 0.923 [0.861, 0.978] | 16.2 | 0/8 |
| streaming | 90.0 [90.0, 90.0] | 0.000 [0.000, 0.000] | 7.5 | 8/8 |

Note: Streaming's `Perfect Grounding` is by-construction; every shipped sentence has passed the per-sentence NLI gate.