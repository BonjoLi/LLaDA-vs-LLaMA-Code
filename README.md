<div id="top"></div>

<h1 align="center">Evaluating Diffusion Language Models Beyond Benchmarks</h1>
<h3 align="center">Human Perception of Generation Dynamics</h3>

<p align="center"><em>How the generation dynamics of diffusion vs. autoregressive LLMs shape user trust, perceived reasoning, and quality.</em></p>

<p align="center">
  <a href="https://anonymous.4open.science/r/LLaDA-vs-LLaMA-Code-8C87">
    <img src="https://img.shields.io/badge/%F0%9F%A4%96%20Code-anonymous.4open.science-grey" />
  </a>
  <img src="https://img.shields.io/badge/%F0%9F%93%9C%20Paper-Under%20Double--Blind%20Review-red" />
  <img src="https://img.shields.io/badge/%F0%9F%A7%AA%20Study-N%3D39%20Within--Subjects-blue" />
</p>

---

## 📝 About

This repository accompanies our paper studying how the **generation dynamics** of large language models influence user perception in an interactive chatbot setting.

Diffusion-based language models (**DLLMs**) now match autoregressive large language models (**ALLMs**) on standard benchmarks and are moving into production. But benchmarks say nothing about how users *experience* these models. Unlike ALLMs, which produce text left-to-right, DLLMs iteratively denoise a noisy text representation into a final output — a visibly distinct process.

We ran a randomized **within-subjects study (N = 39)** in which participants interacted with a DLLM (**LLaDA-8B Instruct**) and a benchmark-comparable ALLM (**LLaMA 3.1-8B Instruct**) through identical chatbot interfaces, then rated each on trustworthiness, reasoning, and quality. We find the **DLLM elicits lower trust and quality ratings**, while the ALLM is rated more trustworthy and draws more anthropomorphic descriptions. This suggests that comparable benchmark performance does **not** guarantee comparable user trust, and that a DLLM's generation dynamic may slow its adoption.

---

## 🔑 Key Findings

Post-interaction ratings (1–7 Likert). The ALLM was rated significantly higher across all three constructs (all *p* < .001).

| Construct        | DLLM (M) | ALLM (M) | Effect size (*d*) |
|------------------|:--------:|:--------:|:-----------------:|
| Trustworthiness  | 4.54     | 5.61     | −0.736            |
| Reasoning        | 4.49     | 5.70     | −0.830            |
| Quality          | 4.35     | 5.63     | −0.835            |

Per-prompt response quality (0–100) followed the same pattern: ALLM **81.26** vs. DLLM **63.20**, with the DLLM also showing higher variability. Differences held after controlling for response latency, response length, LLM familiarity, education level, and presentation order.

---

## ⚖️ Models Compared

Two open-source models, equal in parameter size and similar in benchmarked ability but with distinct generation mechanisms.

| Benchmark  | LLaMA (ALLM) | LLaDA (DLLM) |   Δ    |
|------------|:------------:|:------------:|:------:|
| MMLU       | 69.4%        | 65.5%        | −3.9%  |
| MMLU-pro   | 48.3%        | 37.0%        | −11.3% |
| ARC-C      | 83.4%        | 88.5%        | +5.1%  |

**Decoding configuration**

- **ALLM — LLaMA 3.1-8B Instruct:** autoregressive, nucleus sampling (top-p = 0.9), temperature = 0.2, up to 1024 tokens.
- **DLLM — LLaDA-8B Instruct:** semi-autoregressive block diffusion (gen length = 128, steps = 128, block length = 2), classifier-free guidance (scale = 0.2), low-confidence remasking, temperature = 0.2. Intermediate denoising states revealed at 0.05s/step.

Both models were served through **identically structured Gradio interfaces** to eliminate visual-design bias.

---

## 🧪 Study Design

- **Recruitment:** 54 participants recruited via Prolific; 39 retained after removing incomplete or non-compliant sessions.
- **Design:** within-subjects; participants pseudorandomly assigned to one of two orders (DLLM→ALLM or ALLM→DLLM).
- **Task:** 5 prompts per chatbot. After each response, participants rated quality (0–100); after each full interaction, they gave Likert ratings (1–7) for quality, reasoning, and trustworthiness, plus open-ended responses.
- **Prompts:** ~5,000 prompts filtered from the [Databricks Dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) dataset across five categories — `brainstorming`, `open_qa`, `general_qa`, `creative_writing`, `classification`. Prompts requiring domain knowledge, fixed-context, time-specific (year 2023), or sensitive content were excluded.
- **Compensation:** $5 USD; participants 18+, fluent in English, on a desktop/laptop.

---

## 📁 Directory Structure

> **⚠️ PLACEHOLDER — update to match your actual repo.** This reflects what the paper implies should be here; replace with your real file/folder names.

```
LLaDA-vs-LLaMA-Code/
├── interface/                  # Gradio chatbot interfaces (DLLM + ALLM serving code)
│   ├── dllm_app.py
│   └── allm_app.py
├── data/
│   ├── results_dataset.csv     # De-identified participant responses (N = 39)
│   └── codebook.md             # Variable names, scales, and coding
├── analysis/
│   ├── trust_analysis.jasp     # Linear mixed models, t-tests, order-effects models
│   └── analysis_notes.md       # Maps each test to results in the paper
├── figures/
│   ├── make_figures.py         # Regenerates Figures 2–3 from data/
│   └── *.pdf
├── prompts/
│   └── filtered_prompts.csv    # Curated Dolly-15k subsample
├── requirements.txt
├── LICENSE
└── README.md                   # ← you are here
```

---

## 🚀 Reproducing the Results

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Launch a chatbot interface
python interface/allm_app.py     # or dllm_app.py

# 3. Regenerate figures from the released data
python figures/make_figures.py
```

**Statistics.** Open `analysis/trust_analysis.jasp` in [JASP](https://jasp-stats.org) (free). Each analysis block corresponds to a result in the paper, documented in `analysis/analysis_notes.md`.

---

## 🔒 Ethics & Data

This study involved human participants recruited via Prolific. All participants gave informed written consent. **No personally identifiable information was collected** — only basic demographic items. The released `results_dataset.csv` is fully anonymized and contains only aggregated, de-identified responses used for the analyses in the paper.

---

## 📚 Citation

```bibtex
@inproceedings{anonymous2026diffusion,
  title     = {Evaluating Diffusion Language Models Beyond Benchmarks: Human Perception of Generation Dynamics},
  author    = {Anonymous},
  booktitle = {Workshop on Non-Autoregressive Language Models (NonAR-LM), Conference on Language Modeling (COLM)},
  year      = {2026},
  note      = {Under double-blind review}
}
```

<p align="left"><a href="#top">🔝 Back to Top</a></p>
