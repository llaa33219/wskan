"""Generate Hugging Face model cards (README.md) for the 15 released
wskan-{size}-{dataset} models.

Cards are assembled from the staged artifacts produced by
experiments/RELEASE_publish.py (weights/config/generations/meta), so the
generation outputs embedded in each card are byte-for-byte the real model
outputs. The qualitative evaluations below were written by reading those
outputs; edit them only after re-reading the referenced samples.

Run: .venv/bin/python experiments/RELEASE_modelcard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

STAGING = Path("/tmp/opencode/hf_release")
GITHUB = "https://github.com/llaa33219/wskan"
LOGO = f"{GITHUB.replace('github.com', 'raw.githubusercontent.com')}/main/logo.svg"

SIZES = ["1k", "10k", "100k", "1m", "10m"]
DATASETS = ["tinystories", "ultrachat", "wikitext"]
DATASET_TITLE = {
    "tinystories": "TinyStories",
    "ultrachat": "UltraChat 200k",
    "wikitext": "WikiText-103",
}
DATASET_LINK = {
    "tinystories": "https://huggingface.co/datasets/roneneldan/TinyStories",
    "ultrachat": "https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k",
    "wikitext": "https://huggingface.co/datasets/Salesforce/wikitext",
}

# --- per-model qualitative evaluations (authored from the staged outputs) ---

EVAL = {
("1k", "tinystories"): """At 3,610 parameters this is the capacity floor of the family, and the outputs show it.
The continuation is orthographic noise with English-like letter statistics: real function words
("the", "she", "and") surface inside otherwise invented strings ("Thed Jlaegy", "tannedgd atrmosed").
The decisive observation is that **all three prompts converge to the same continuation** — the model
has collapsed onto a prompt-insensitive attractor, meaning unconditional byte statistics dominate any
conditioning. It has learned the surface rhythm of TinyStories (short sentences, quotation marks) but
neither vocabulary nor grammar. Useful as a lower bound for scaling studies, not as a text generator.""",

("10k", "tinystories"): """Real words now appear frequently ("she", "mom", "said", "a bird") but are interleaved with
near-miss pseudowords ("dod", "stogetty", "swaler", "nicle"). Story-opening conventions and dialogue
quotes emerge (`said, "No, ...`), yet syntax collapses within a few words and the continuations still
slide toward shared attractor text across prompts. The model has learned vocabulary-shaped byte
statistics and the cadence of children's stories, but not yet reliable word formation or grammar.
Compared to the 1k tier the attractor is weaker and prompt conditioning is visible in the first few
words, but this tier is still best understood as a stylometric mimic.""",

("100k", "tinystories"): """A qualitative jump: fully grammatical, correctly spelled English with recognizable
story structure — openings, character names, dialogue, even a spontaneous "The end." followed by a new
story. Prompt conditioning works: each prompt receives a distinct, on-theme continuation. What remains
loose is semantics: objects and predicates mismatch ("a blanket delicious horn", "found a bigger who
smiled"), and narratives drift between unrelated storylets within a few sentences. At the sentence
level this is nearly indistinguishable from the TinyStories register; beyond 2-3 sentences, coherence
is not guaranteed. This tier is where the architecture first demonstrates real language modeling rather
than byte statistics.""",

("1m", "tinystories"): """Sentences are grammatical and locally consistent, with concrete imagery ("It was red
and had many leaves on it") and working dialogue attribution. Multi-sentence coherence now holds for
roughly 3-5 sentences before topic drift sets in. The remaining errors are semantic rather than
grammatical — a blanket that is simultaneously "green and it was big and rain" — and entity tracking
across a paragraph is unreliable (Tom speaks, then "she" follows the box). Story beats (problem,
reaction, resolution) appear spontaneously. A competent stylistic model of children's narrative.""",

("10m", "tinystories"): """The strongest model of the family produces coherent micro-stories with setup,
dialogue, and resolution structure, and pronouns track referents across sentences
("He missed her and said, 'I'm sorry, mommy.'"). Prompts are honored: the "lost his" prompt yields
"lost his favorite toy." Occasional nonsense objects persist ("a thoughtful lion", "what's behind the
barber") and story logic stays dream-like, but the output is recognizably TinyStories-style children's
narrative throughout. Keep in mind this is still a 10M-parameter byte-level model: factual accuracy
and long-range consistency beyond a paragraph should not be expected.""",

("1k", "ultrachat"): """At 3,610 parameters the model cannot represent the UltraChat distribution: the
continuation is prompt-insensitive gibberish, **byte-identical across all three prompts** — including
the factual question, the explanation request, and the poem request. The "Assistant:" marker is
echoed only because it is in the prompt; nothing about the response is conversational. Word-shaped
fragments carry a faint English texture but no real words beyond chance. This tier serves as the
capacity floor: it shows what the architecture does when parameters are the binding constraint.""",

("10k", "ultrachat"): """The conversational surface structure arrives before the content: the model reproduces
the Assistant-turn format and instructional-text furniture — numbered lists ("1.", "3."), imperative
fragments ("provide", "impact") — but the words themselves are mostly pseudowords ("resicusion",
"tandedgect", "innerstant"). No question is answered; the poem request gets list-shaped noise.
What has been learned at this tier is the *genre skeleton* of UltraChat (turns, lists, advice
register) with word formation still lagging behind.""",

("100k", "ultrachat"): """Fluent, correctly-spelled English that completely fails to engage with the prompt:
the capital-of-France question receives a drifting expository paragraph touching politics, pasta, and
population; the photosynthesis question receives something about construction and social media. The
model has thoroughly learned UltraChat's *register* — list markers, section labels ("Scene 3:"),
recipe-like fragments, hedge phrases ("However,", "including") — but treats the prompt as a genre
trigger rather than a query. This is the clearest illustration in the family of the gap between
stylistic modeling and instruction following: all of the former, none of the latter.""",

("1m", "ultrachat"): """Assistant-persona phrases now appear spontaneously ("Yes, I can tell...",
"Here are a few things you must add"), and the opening clause of each response is topically adjacent
to the prompt before drifting. The photosynthesis answer gestures at "correlation between men's
physics and feeding" — wrong, but recognizably an attempt at an explanation register. Multi-turn
format is learned (a follow-up "User:" turn is generated after the answer). Grounding remains absent:
no factual question is answered correctly. Tone and format: solved. Content: not yet.""",

("10m", "ultrachat"): """The most conversational model of the family, and the most instructive failure. It
sustains the User:/Assistant: turn-taking format perfectly — after completing an answer it generates
the *next* user turn, exactly matching the training render format, and role-plays both sides
coherently. The prose is fluent, structured, and persona-consistent. But the answers do not address
the questions: the France question gets a paragraph about a "Dali Director", and the poem request gets
a friendly prose reply about writing instead of a poem. Instruction following is not learned at 10M
parameters; conversational *form* is fully learned. Use this model to study chat format emergence,
not as a chatbot.""",

("1k", "wikitext"): """Byte-statistical noise, but the dataset fingerprint is already visible: WikiText's
signature spaced punctuation (" , " " .") pervades the output, and proper-noun-shaped capitalized
fragments appear ("Bers", "Jagecath", "Kedes"). As with the other 1k tiers, the continuation is a
prompt-insensitive attractor. Nothing encyclopedic is present — the model has captured only the
typography of WikiText-103.""",

("10k", "wikitext"): """WikiText's surface conventions emerge ahead of grammar: spaced punctuation, section
fragments ("= ="), date-like tokens ("2005"), and capitalized entity-shaped strings ("Asaitian
Starkin Lawan"). Sentences are not grammatical, but the output is unmistakably *trying* to be an
encyclopedia article — the byte statistics of headings, parentheticals, and dates are learned before
words are. Reads as a wiki page rendered in a dream.""",

("100k", "wikitext"): """Grammatical encyclopedic prose with WikiText markup habits intact: "@-@" number
formatting, spaced punctuation, parentheticals, battle/film/biography registers. The content is
entirely confabulated — plausible-sounding battles, places, and dates with no factual basis
("Quimanánso layers", "Horright Harbour"). The genre is captured convincingly; the knowledge is
absent. A useful demonstration that byte-level LMs at this scale model *style* far earlier than
*facts*.""",

("1m", "wikitext"): """Article-like structure with parenthetical asides, em-dashes, and consistently formatted
invented entities ("the Jagdca Grosman", "the Panic 's business"). Sentences are grammatical and the
first sentence after the prompt stays on-register ("Ben Daniel , leading element of religious
captain"). Topical consistency degrades over a paragraph, and every fact is fabricated. The model is
a reliable WikiText *stylist*, an unreliable encyclopedia.""",

("10m", "wikitext"): """A strong WikiText-103 imitation: the film prompt is answered in-register
("Ben Daniel ."), followed by a correctly formatted section heading ("= William R. Burns ="),
vital-date parentheticals ("( 1917 – 1920 )"), and the dataset's "@-@" conventions
("127 @.@ 5 miles"). Multi-sentence expository structure holds better than at any smaller tier.
Every specific fact is confabulated — names, dates, and places are invented — so treat this strictly
as a stylistic model of the encyclopedia, never as a knowledge source.""",
}

SIZE_BLURB = {
    "1k": "capacity floor of the family",
    "10k": "word-formation tier",
    "100k": "fluency tier",
    "1m": "local-coherence tier",
    "10m": "flagship tier",
}

FEATURES = """- **Kolmogorov-Arnold Network**: learnable univariate *edge functions* replace scalar weights.
- **Wavelet-like SSM edges**: each edge function is the impulse response of a stable state-space
  model — a damped oscillator (decay sigma, frequency omega) that acts as a wavelet-like kernel.
- **Fully selective recurrence**: input-dependent step size dt and input/output projections (B, C),
  so the wavelet transform is content-warped per token (Mamba-style selectivity).
- **Fused Triton scan**: the associative scan is a single fused GPU kernel (log-depth), verified
  against the reference scan (fwd err 7e-7, grads <= 4e-6).
- **Byte-level**: vocabulary is the 256 byte values — no tokenizer, no BPE, no special tokens.
- **Interpretable by construction**: every edge exposes its learned (sigma, omega) spectrum; see the
  interpretation reports in the GitHub repo."""

LIMITATIONS = """- **Research artifact.** These are 3.6k-10M parameter proof-of-concept models from an
  architecture study, not production LMs.
- **No factual reliability.** All models confabulate names, dates, and facts; the larger ones merely
  do so more fluently. Never use outputs as a source of truth.
- **Not instruction-tuned.** Even the UltraChat models only imitate conversational *format*; they do
  not follow instructions.
- **Short-range memory.** The interpretation reports show the models' effective memory is local
  (dozens of bytes); long-range consistency is not to be expected.
- **English-only, single-domain.** Each model knows only its training slice.
- **CUDA GPU required for inference.** The recurrent scan is a fused Triton kernel; there is no CPU
  path. (`config.yaml: architecture.bf16_scan: false` switches the scan to fp32 on GPU.)
- **Sampling only.** Generation is multinomial sampling; outputs vary with seed and temperature."""

COLLECTION_NOTE = {
    "tinystories": "children's stories (roneneldan/TinyStories)",
    "ultrachat": "rendered chat conversations (HuggingFaceH4/ultrachat_200k train_sft)",
    "wikitext": "encyclopedic prose (Salesforce/wikitext wikitext-103-raw-v1)",
}


def collection_table(current: str) -> str:
    rows = ["| Model | Params | Dataset | Final eval loss | Byte PPL |",
            "|---|---|---|---|---|"]
    for ds in DATASETS:
        for size in SIZES:
            name = f"wskan-{size}-{ds}"
            meta = json.loads((STAGING / name / "meta.json").read_text())
            mark = " **(this model)**" if name == current else ""
            rows.append(
                f"| [{name}](https://huggingface.co/llaa33219/{name}){mark} "
                f"| {meta['params']:,} | {DATASET_TITLE[ds]} | {meta['eval_loss']:.4f} "
                f"| {meta['eval_byte_ppl']} |"
            )
    return "\n".join(rows)


def card(name: str) -> str:
    size, dataset = name.split("-", 1)[1].split("-", 1)
    meta = json.loads((STAGING / name / "meta.json").read_text())
    cfg = yaml.safe_load((STAGING / name / "config.yaml").read_text())
    gens = json.loads((STAGING / name / "generations.json").read_text())
    arch = cfg["architecture"]
    train = cfg["training"]

    samples = []
    for g in gens:
        samples.append(f"**Prompt:**\n```\n{g['prompt']}\n```\n**Output (seed {g['seed']}, "
                       f"temperature {g['temperature']}, {g['max_new']} new bytes):**\n```\n{g['output']}\n```")
    samples_md = "\n\n".join(samples)
    config_yaml = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)

    return f"""---
pipeline_tag: text-generation
tags:
- wskan
- kolmogorov-arnold-network
- state-space-model
- wavelet
- byte-level
- research
---

<p align="center">
  <img src="{LOGO}" alt="WSKAN logo" width="600">
</p>

# {name}

**{meta['params']:,} parameters · WSKAN-11 · byte-level LM trained on [{DATASET_TITLE[dataset]}]({DATASET_LINK[dataset]})**
({SIZE_BLURB[size]})

## Introduction

`{name}` is a checkpoint of **WSKAN-11** (Wavelet-like State-Space KAN, version 11), a
Kolmogorov-Arnold Network in which every edge function is the impulse response of a stable
state-space model — a wavelet-like damped oscillator. One parameterization supports two modes: a
closed-form wavelet edge (static, WavKAN-style) and a recurrent SSM edge (Mamba-style selective
scan); this checkpoint runs the recurrent mode through a fused Triton associative-scan kernel.

This model is one cell of the 3-epoch release matrix (5 sizes x 3 datasets, seed 42) trained for the
WSKAN architecture study. It is released to make the study's qualitative claims directly
inspectable: the generation results below are real, unedited outputs of these exact weights.

- **GitHub repository (code, reports, training scripts):** {GITHUB}
- **Architecture document:** [models/V11_README.md]({GITHUB}/blob/main/models/V11_README.md)
- **Definitive interpretation report:** [experiments/W11_FINAL_INTERPRETATION_REPORT.md]({GITHUB}/blob/main/experiments/W11_FINAL_INTERPRETATION_REPORT.md)

## Features

{FEATURES}

## Model collection

{collection_table(name)}

This model is trained on {COLLECTION_NOTE[dataset]}; the other rows cover the remaining two domains
and sizes. All rows use the identical WSKAN-11 architecture, differing only in depth/width.

## Generation results (real outputs)

The following are **actual outputs of this checkpoint**, generated with
`experiments/RELEASE_infer.py` from the released weights (multinomial sampling, temperature 0.8,
seed 1234, 400 new bytes). Nothing is cherry-picked or edited — these are the first and only samples
drawn for this card.

{samples_md}

### Evaluation of the outputs

{EVAL[(size, dataset)]}

Objective metrics for this checkpoint: final eval loss **{meta['eval_loss']:.4f}**
(byte-level perplexity **{meta['eval_byte_ppl']}**) on a held-out slice of
{DATASET_TITLE[dataset]}, at training step {meta['ckpt_step']:,}.

## Configuration

The released `config.yaml` (verbatim):

```yaml
{config_yaml}```

Architecture quick reference: d_model={arch['d_model']}, n_layers={arch['n_layers']},
n_states={arch['n_states']}, bc_rank={arch['bc_rank']}, vocab=256 (raw bytes).
Training: {train['steps']:,} steps, batch {train['batch_size']}, block {train['block_size']},
lr {train['lr']} (cosine), AdamW, seed {train['seed']}, ~3 epochs, bf16 scan.

## Local run

Inference requires an NVIDIA GPU with Triton (the scan kernel is fused Triton; there is no CPU
path) and Python 3.12+.

```bash
git clone {GITHUB}.git
cd wskan
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python safetensors pyyaml huggingface_hub

# generate from this model (downloads config.yaml + model.safetensors from the Hub)
.venv/bin/python experiments/RELEASE_infer.py \\
    --model llaa33219/{name} \\
    --prompt "{gens[0]['prompt'].splitlines()[0]}" \\
    --max-new 400 --temperature 0.8 --seed 1234
```

`--model` also accepts a local directory containing `config.yaml` and `model.safetensors`.
Weights are fp32; set `bf16_scan: false` in `config.yaml` for an fp32 scan (outputs will differ
slightly from the samples above, which used the training-time bf16 scan).

To retrain from scratch under the identical protocol:

```bash
uv pip install --python .venv/bin/python datasets transformers
.venv/bin/python experiments/V1_train_tinystories_lm.py \\
    --model wskan11 --scale {size} --dataset {dataset} --seed 42 \\
    --batch {train['batch_size']} --block {train['block_size']} \\
    --steps {train['steps']} --lr {train['lr']} --lr-schedule cosine \\
    --ckpt-every 25000 --eval-every 1000 --out-tag 3ep --compile --bf16
```

## Limitations

{LIMITATIONS}

## Provenance

- Checkpoint: `checkpoints/wskan11_{dataset}_{size}_3ep_s42/latest.pt` (step {meta['ckpt_step']:,})
  of the WSKAN repository, converted to safetensors (fp32) without any weight modification.
- Training data: [{DATASET_TITLE[dataset]}]({DATASET_LINK[dataset]}), byte-encoded UTF-8.
- Seed 42 is the canonical seed of the 5-seed campaign; the released sample outputs above used
  sampling seed 1234.
"""


def main() -> None:
    for dataset in DATASETS:
        for size in SIZES:
            name = f"wskan-{size}-{dataset}"
            (STAGING / name / "README.md").write_text(card(name))
            print("card written:", name)


if __name__ == "__main__":
    main()
