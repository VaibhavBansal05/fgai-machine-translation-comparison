# Results and discussion

The measured table and score plot are in [results/summary.md](results/summary.md) and [results/scores.svg](results/scores.svg). All four systems used the same 20 held-out English-to-French pairs, greedy decoding, and a 40-token output cap on a CPU. The scratch Transformer and mT5 each trained on 200 pairs; TinyLlama and Gemma used the same zero-shot prompt. These settings make the project quick to reproduce but do not produce a reliable production translator.

## What the outputs show

- **Basic Transformer:** BLEU 0.00 and chrF 0.27. Its test outputs are largely repeated hyphens. Training from scratch on 200 literary pairs is far too little for a useful word-level translator.
- **mT5:** BLEU 0.27 and chrF 2.89. All 20 outputs contain `<extra_id_0>`, a pretraining sentinel. A single epoch on 200 pairs did not adapt the model to translation. This is a clear failure, not a usable French translation system.
- **TinyLlama:** BLEU 16.31 and chrF 40.95, the highest in this run. It translates many examples but changes meaning in some. For “There were four,” it wrote “Il y a quatre,” changing past tense to present. For the Bonacieux sentence it changed the active subject into a passive construction. It also rendered “The laughter began again” as a sentence about warmth, which is a meaning substitution.
- **Gemma:** BLEU 6.93 and chrF 30.74. It copied the English source verbatim in 7 of 20 cases, including “There were four.” Other outputs mix English and French. Its small size and prompt-only method limit this comparison.

The two learned models had training times; the prompted models had none. TinyLlama is much larger than Gemma, mT5, and the scratch model. Thus the observed ranking combines architecture, pretraining, tokenizer, size, and adaptation method. It does **not** prove that decoder-only architecture is intrinsically better. The source corpus also has some noisy alignments: the test reference for “She would see the cat...” contains details absent from the English source. Automatic scores are therefore imperfect evidence; the CSV outputs should be read alongside them.

## Sampling temperature

The prompted-model CSVs include two source sentences at temperatures 0, 0.7, and 1.0. TinyLlama's “There were four” output changes from “Il y a quatre” at temperature 0 to “Encore quatre!” at 1.0, which drops the original tense and sentence structure. Gemma copied the same sentence in English at all three temperatures. Raising temperature did not reliably fix translation and sometimes increased meaning drift.

## Responsible use

The observed tense, subject, and content changes illustrate hallucination or semantic distortion. The literary corpus may also carry dated cultural language and uneven coverage of names and places. Such errors could be harmful when translating sensitive medical, legal, or safety information. Gendered terms, names, numbers, negation, cultural expressions, and potentially offensive source text need particular care. This small test set does not measure demographic bias or sensitive-content safety directly, so no safety claim is made.

For safer use, show the original beside the translation, require human review for consequential content, check named entities and numbers automatically, test on diverse contemporary examples, report both BLEU/chrF and human error categories, and provide a way to correct or reject translations. Do not deploy these experimental checkpoints as autonomous translators.

**Conclusion:** TinyLlama has the best measured chrF and BLEU in this run. Its translations still need review, and a larger, cleaner training set would be required for the two supervised models to become useful.
