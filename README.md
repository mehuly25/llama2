# Activation Steering in Llama 2 7B for Bilingual Bias Detection

> A multilingual, representation-level study of social bias and refusal behaviour in LLMs using Contrastive Activation Steering.

---

## Overview

Large language models (LLMs) are increasingly used in systems that support or shape decision-making — making it essential to understand whether they reproduce harmful social biases present in human-generated data.

This project investigates **bias and refusal behaviour** in `Llama 2 7B Chat` and `GPT-OSS-20B` through the lens of **Contrastive Activation Steering (CAS)**. Rather than analysing surface-level outputs alone, we probe the models' internal activation space to examine how bias is encoded at the representation level — and whether safety mechanisms such as RLHF truly eliminate biased representations, or merely suppress them.

A key contribution of this work is its **bilingual design**: all experiments are conducted in both **English and Italian**, enabling cross-linguistic comparison of bias expression and model behaviour.

### Key Questions

- Can stereotype and anti-stereotype prompts be separated in the model's activation space?
- Does removing the refusal vector reveal biased completions that safety behaviour would otherwise suppress?
- Do bias directions generalise across domains (gender, race, religion), models, and languages?
- Does RLHF reduce bias, or does it make different forms of bias harder to distinguish?

---

## Methodology

### Contrastive Activation Steering (CAS)

We use CAS to build **steering vectors** for four directions:

| Steering Vector | Description |
|---|---|
| `gender` | Gender bias direction |
| `race` | Racial bias direction |
| `religion` | Religious bias direction |
| `refusal` | Model refusal/safety behaviour direction |

Steering vectors are computed as the mean difference between activations produced by **stereotype** and **anti-stereotype** prompt pairs, across multiple model layers. At inference time, these vectors are injected into the residual stream to steer model behaviour.

### Experimental Conditions

For each prompt, we compare three output types:

1. **Un-steered** — baseline model output
2. **Bias-only steered** — bias vector applied (e.g. gender steering)
3. **Combined bias + refusal steered** — both bias and refusal vectors applied simultaneously

---

## Datasets

### English Dataset (`Dataset/`)

The English dataset is composed of two sources:

**StereoSet**
A widely used benchmark for measuring stereotypical bias in language models. It provides paired stereotype and anti-stereotype sentences across gender, race, religion, and profession domains. We use the intersentence split, which requires the model to complete a context with a stereotypical or anti-stereotypical continuation.

**Custom GPT-4 Prompts**
To supplement StereoSet and improve coverage, we generated additional bias prompts using GPT-4. These prompts are designed to elicit stereotype and anti-stereotype activations in domains and formulations not well represented in StereoSet, while maintaining the paired contrastive structure required for CAS.

### Italian Dataset (`Dataset_Italian/`)

A parallel Italian version of the full dataset was created through **machine translation**, followed by **human verification**. This ensures that the Italian prompts are linguistically natural and semantically equivalent to their English counterparts — not just literal translations.

The Italian dataset enables direct cross-linguistic comparison and allows us to test whether bias directions and refusal behaviours generalise across languages.

---

## Repository Structure

```
llama2/
│
├── Dataset/                        # English prompts (StereoSet + GPT-4 custom)
├── Dataset_Italian/                # Italian prompts (machine-translated + human-verified)
│
├── dataset_translation_italian.ipynb       # Translates English dataset to Italian
│
├── generate_steering_vectors.ipynb         # Builds steering vectors (English)
├── generating_steering_vectors_IT.ipynb    # Builds steering vectors (Italian)
│
├── get_steered_responses.ipynb             # Generates steered outputs (English)
├── get_steered_responses_IT.ipynb          # Generates steered outputs (Italian)
├── get_steered_responses_new.ipynb         # Updated steered responses (English)
├── get_steered_responses_new_IT.ipynb      # Updated steered responses (Italian)
│
├── refusal_steering.ipynb                  # Refusal vector experiments (English)
├── refusal_steering_IT.ipynb               # Refusal vector experiments (Italian)
│
├── visualise_vectors.ipynb                 # PCA/clustering visualisation of vectors
│
├── responses/                      # English model outputs (un-steered)
├── responses_IT/                   # Italian model outputs (un-steered)
├── responses_new_eng/              # Updated English outputs
├── responses_new_IT/               # Updated Italian outputs
├── responses_xls/                  # Outputs in Excel format for analysis
│
├── gender_rlhf/                    # Gender steering results (English)
├── gender_rlhf_IT/                 # Gender steering results (Italian)
├── race_rlhf/                      # Race steering results (English)
├── race_rlhf_IT/                   # Race steering results (Italian)
├── religion_rlhf/                  # Religion steering results (English)
├── religion_rlhf_IT/               # Religion steering results (Italian)
│
├── refusal_responses/              # Refusal steering outputs (English)
├── refusal_responses_IT/           # Refusal steering outputs (Italian)
├── refusal_rlhf/                   # Refusal + bias combined results (English)
├── refusal_rlhf_IT/                # Refusal + bias combined results (Italian)
│
└── clustering/                     # Clustering analysis of activation vectors
```

---

## Main Findings

- **Evidence of social bias in both models and both languages**, with especially clear effects for **gender** and **religion**.
- **Cross-linguistic variation**: bias is not expressed uniformly across languages, even for semantically equivalent prompts.
- **Language switching**: models sometimes switch from Italian to English when responding to sensitive prompts.
- **Refusal does not imply absence of bias**: biased completions can resurface when the refusal vector is removed via activation steering, suggesting that safety mechanisms suppress rather than eliminate biased representations.
- **RLHF homogenises bias representations**: post-RLHF models show more similar internal representations across different forms of societal bias, raising questions about the model's ability to distinguish between them.

---

## Requirements

The notebooks are self-contained and run in a standard Python environment with access to a GPU. Main dependencies:

- `torch`
- `transformers`
- `numpy`, `pandas`
- `matplotlib`, `seaborn`
- `scikit-learn`
- Hugging Face access token for `meta-llama/Llama-2-7b-chat-hf`

---

## Citation

If you use this code or dataset in your work, please cite:

```bibtex
@inproceedings{your_paper_2026,
  title     = {Activation Steering in Llama 2 7B for Bilingual Bias Detection},
  author    = {[Mehuly Chakraborthy]},
  booktitle = {To be updated},
  year      = {2026}
}
```

---

## Notes on AIES Submission

This repository accompanies a paper submitted to **AIES 2026** (9th AAAI/ACM Conference on AI, Ethics, and Society). GPT-4 was used exclusively to generate custom bias prompts as part of the experimental data — not to write any portion of the paper text.

---

## Based On

This project builds on the code and methodology from:

> **Investigating Bias Representations in Llama 2 Chat via Activation Steering**
> Dawn Lu (UC Berkeley) & Nina Rimsky (SPAR)
> arXiv:2402.00402v1, February 2024
> [https://arxiv.org/abs/2402.00402](https://arxiv.org/abs/2402.00402)

The original work applies Contrastive Activation Steering to study bias in Llama 2 using **English-only** prompts. This project extends their framework by introducing a **parallel Italian dataset** (machine-translated and human-verified), enabling the first cross-linguistic comparison of bias representations and refusal behaviour under activation steering.
---

## License

This project is for research purposes. Please refer to the original [StereoSet](https://github.com/moinnadeem/StereoSet) license for dataset usage terms.
