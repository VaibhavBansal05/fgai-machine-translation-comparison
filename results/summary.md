# Measured results

All models use the same English-to-French test set and greedy decoding (temperature 0).

| Model | Adaptation | BLEU | chrF | Train seconds | Seconds/sentence | Parameters |
|---|---|---:|---:|---:|---:|---:|
| basic | trained | 0.00 | 0.27 | 1.5 | 0.105 | 1,138,715 |
| mt5 | trained | 0.27 | 2.89 | 90.6 | 0.072 | 300,176,768 |
| llama | prompt only | 16.31 | 40.95 | N/A | 2.563 | 1,100,048,384 |
| gemma | prompt only | 6.93 | 30.74 | N/A | 0.520 | 268,098,176 |

Best chrF on this test set: **llama**. BLEU, speed, and parameter count may favor other models.

Prompt-only models have no task training time. Timing depends on this run's hardware and tokenizer. Parameter count and pretraining differ, so architectural effects cannot be isolated.
