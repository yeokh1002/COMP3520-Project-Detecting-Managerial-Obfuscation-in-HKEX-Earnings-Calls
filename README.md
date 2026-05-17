# Detecting Managerial Obfuscation in HKEX Earnings Calls

A fine-tuned NLP pipeline for detecting **managerial obfuscation** — the strategic use of vague, complex, or jargon-heavy language to obscure substantive content — in earnings conference calls of Hong Kong Stock Exchange (HKEX) listed firms, and an empirical study of its relationship with post-earnings stock returns.

This repository accompanies the COMP3520 (HKU) final project. It contains the three Jupyter notebooks that, end-to-end, (1) generate obfuscation labels for earnings-call excerpts via DeepSeek-Chat, (2) fine-tune a hybrid FinBERT classifier on those labels, and (3) score full Q&A transcripts and analyse the resulting obfuscation scores against post-earnings stock returns.

---

## Background

Most financial NLP focuses on **sentiment polarity**: is management's tone positive or negative? But the Managerial Obfuscation Hypothesis (Bloomfield, 2002; Li, 2008) argues that what managers *don't* say can matter more than the tonal surface — executives with bad news have strong incentives to keep language positive-sounding while withholding specifics. A textbook example: Seagate Technology's April 2017 earnings call featured a heavily hedged, jargon-laden Q&A response that was followed by a 30.1% stock decline over the next five months as the underlying issues surfaced.

This project sets out to:

1. Build the first dedicated obfuscation-labelled dataset for HKEX earnings calls (3,000+ excerpts, 200+ transcripts, 2023–2025).
2. Fine-tune a transformer-based classifier (FinBERT) to detect obfuscation directly, augmented with linguistic features grounded in the Loughran–McDonald dictionary and standard readability metrics.
3. Test whether the resulting obfuscation signal predicts post-earnings stock returns in the Hong Kong market.

The final classifier reaches **F1 = 0.824** and **recall = 0.835** on a held-out test set. The downstream financial analysis produces results that depart from the canonical U.S.-based MOH prediction in institutionally interesting ways — see *Key Findings* below.

---

## Repository Contents

| Notebook | Purpose |
| --- | --- |
| `Deepseek_labelling.ipynb` | Labels earnings-call excerpts using DeepSeek-Chat as a rubric-prompted annotator (5 dimensions × continuous [0,1] scoring, summed and thresholded at 2.5). Produces the binary labels used downstream. |
| `FinBERT_FineTuning_Full.ipynb` | Fine-tunes FinBERT (`yiyanghkust/finbert-pretrain`) on the DeepSeek labels. Trains five model variants in an ablation: raw FinBERT, vanilla fine-tuned, FinBERT + basic features, FinBERT + extended features, and the final variant adding FGM adversarial training. |
| `Full_Obfus_VS_Return_Notebook.ipynb` | Applies the trained classifier to 204 full Q&A transcripts, computes firm-level obfuscation scores, and analyses correlations, quintile portfolios, and sentiment × obfuscation "gap" portfolios against same-day, overnight, 1-day, 5-day, 20-day, 50-day, 100-day, and 250-day nominal and HSI-adjusted abnormal returns. |

---

## Methodology Pipeline

```
Earnings-call transcripts (HKEX, 2023–2025)
         │
         ▼  segment + clean
Excerpts (~50–150 words each)
         │
         ▼  Deepseek_labelling.ipynb
DeepSeek-Chat (temperature = 0.0) scores 5 dimensions
   • Numerical Specificity
   • Quantification
   • Jargon
   • Hedging
   • Actionability
sum ∈ [0,5];  binary label: 1 if sum > 2.5 else 0
         │
         ▼  manual verification on sampled subset
Labelled dataset (3,000+ excerpts)
         │
         ▼  FinBERT_FineTuning_Full.ipynb
Fine-tuned hybrid FinBERT
   = BERT [CLS] (768-d)  ⊕  linguistic features (12-d)
   + Fast Gradient Method (FGM) adversarial training
         │
         ▼  Full_Obfus_VS_Return_Notebook.ipynb
Score 204 full Q&A transcripts → obfuscation score ∈ [0,1]
         │
         ▼
Correlation / quintile / gap analysis vs. stock returns
```

### The Five-Dimension Labelling Rubric

| Dimension | Score 0 (Clear) | Score 0.5 (Mixed) | Score 1 (Obfuscatory) |
| --- | --- | --- | --- |
| Numerical Specificity | "Revenue +12%", "EPS $1.22–1.28" | "High single-digit growth", "double-digit increase" | "Strong growth", "meaningful traction" |
| Quantification | "$1.2B revenue", "15M users", "350 bps" | "Up from last year", "margin improved" | "Strong performance", "well-positioned" |
| Jargon | Plain language | "Achieved synergies of $50M" | "Strategic initiatives", "ecosystem", "tailwinds" |
| Hedging | "will", "are", "did" | "expect to", "anticipate", "likely" | "we think", "potentially", "if conditions permit" |
| Actionability | "We launched X", "invested $X to achieve Y" | "We are investing", "focusing on" | "Momentum", "strength", "positioning" |

### The Twelve Linguistic Features (Hybrid Model)

Concatenated with the 768-dimensional [CLS] embedding before the classification head: Gunning Fog Index, SMOG Index, average sentence length, polysyllabic word ratio, LM uncertainty %, LM litigious %, modal dissonance (weak − strong), passive-voice ratio, nominalisation density, average word length, word count, sentence count.

---

## Results

### Classification Performance (held-out test set)

| Model | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Raw FinBERT (zero-shot) | 0.511 | 0.512 | 0.939 | 0.663 |
| Vanilla fine-tuned FinBERT | 0.796 | 0.800 | 0.800 | 0.800 |
| FinBERT + basic features (8) | 0.818 | 0.830 | 0.809 | 0.819 |
| FinBERT + extended features (12) | 0.813 | 0.817 | 0.817 | 0.817 |
| **Final: extended features + FGM** | **0.818** | **0.814** | **0.835** | **0.824** |

### Key Findings

- **The classifier works.** It exceeds the >0.80 F1 target and gives a sanity-check obfuscation score of 0.919 on the canonical Seagate Q3-2017 obfuscatory passage (no Seagate material in training).
- **The MOH prediction does not hold straightforwardly in the HK market.** The only statistically significant return correlation is a *negative* same-day association (Pearson r = −0.156, p = 0.026), which we read as reverse causation — managers adapt language in response to within-day price action rather than the reverse.
- **At longer horizons the pattern reverses.** The clearest quintile (Q1) underperforms the most obfuscatory quintile (Q5) by ~16.9 percentage points in nominal returns and ~12.5 percentage points in HSI-adjusted abnormal returns over a 250-day horizon. We interpret this as evidence of an "over-promising" channel: clarity sets expectations that subsequent results fail to validate.
- **Gap analysis is directionally consistent with MOH but underpowered.** Positive + High-Obfuscation calls underperform Neutral + Low-Obfuscation calls by ~4 percentage points over 5 days (p = 0.294).

See the *Final Report* for full discussion.

---

## How to Run

Each notebook is fully self-contained and was developed on Google Colab's free T4 GPU. Open in Colab and run cells top-to-bottom.

### Prerequisites

- **Python 3.10+** (Colab default works)
- Key packages: `transformers`, `torch`, `pandas`, `numpy`, `scikit-learn`, `nltk`, `textstat`, `yfinance`, `scipy`, `matplotlib`, `seaborn`, `openai` (for the DeepSeek API client)
- **A DeepSeek API key** if re-running the labelling notebook (`Deepseek_labelling.ipynb`). The fine-tuning and analysis notebooks do not require any API key.

### Running Order

1. `Deepseek_labelling.ipynb` — produces the labelled CSV (skip if you already have labelled data)
2. `FinBERT_FineTuning_Full.ipynb` — trains the classifier; saves model checkpoint
3. `Full_Obfus_VS_Return_Notebook.ipynb` — loads the checkpoint and runs the financial analysis

### Reproducibility

Random seed = 50 across NumPy, PyTorch, Python hash seed, and CUDA back-ends. `torch.backends.cudnn.deterministic = True`. Train/val/test split uses `random_state = 42` with stratification on the binary label. DeepSeek API calls use `temperature = 0.0`. The full pipeline trains end-to-end in under 30 minutes on a T4 GPU.

---

## Model Configuration

| Hyperparameter | Value |
| --- | --- |
| Backbone | `yiyanghkust/finbert-pretrain` |
| Learning rate | 2e-6 |
| Batch size | 8 |
| Epochs | 5 (with early stopping, patience = 2 on val F1) |
| Max sequence length | 128 tokens |
| Weight decay | 0.05 |
| Optimiser | AdamW |
| Dropout | 0.1 |
| FGM ε (adversarial perturbation) | 0.1 |

---

## Acknowledgements

This project was developed for COMP3520 (Special Topics in Data Science), The University of Hong Kong, supervised by Prof. Huang Chao. The classifier backbone is FinBERT (Yang, Uy, & Huang, 2020). The labelling pipeline uses DeepSeek-Chat (`deepseek-chat`). The financial dictionary is the Loughran–McDonald 2018 master edition (Loughran & McDonald, 2011). HSI return data is from Yahoo Finance via `yfinance`. The LLM-as-annotator methodology draws on Gilardi, Alizadeh, and Kubli (2023).

## Authors

- Yeo Kiah Huah 
- Deng Decheng 

## Disclaimer

This repository is for academic and research purposes only. The obfuscation scores produced by this model are not investment advice. The findings should not be assumed to generalise without further testing.
