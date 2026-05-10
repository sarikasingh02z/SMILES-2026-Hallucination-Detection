# SOLUTION.md — Hallucination Detection via Hidden-State Probing

## How to Reproduce My Results

**Environment:** Python 3.10+, Google Colab T4 GPU (free tier)

```bash
git clone https://github.com/sarikasingh02z/SMILES-2026-Hallucination-Detection.git
cd SMILES-2026-Hallucination-Detection
pip install -r requirements.txt
python solution.py
```

That's it. Running `solution.py` generates both `results.json` and `predictions.csv` automatically. All random seeds are fixed at 42 so results should be identical every run.

---

## What I Built and Why

### The basic idea

The task is to figure out whether Qwen2.5-0.5B is hallucinating or telling the truth — just by looking at its internal activations, not the text itself. That's actually a pretty cool problem because it means the model's hidden states carry some signal about its own confidence or "knowledge state."

### aggregation.py — Getting features out of the model

The model has 24 layers. Each layer produces a hidden state for every token in the input. I needed to collapse all of that into one flat vector per sample.

I went with the simplest approach that made conceptual sense: take the last real token from the final layer. In decoder-only transformers like Qwen, the last token attends to everything before it — it's essentially the model's compressed summary of the entire input. So it felt like the right place to look for hallucination signal.

This gives a 896-dimensional vector per sample, which is manageable.

### splitting.py — Train/val/test split

Stratified split to preserve the class ratio (70% hallucinated, 30% truthful) across all three subsets:
- 70% train (481 samples)
- 15% val (104 samples)
- 15% test (104 samples)

Stratification matters here because the dataset is imbalanced — without it you could end up with splits that don't reflect the real distribution.

### probe.py — The actual classifier

I used a small 2-layer MLP:
`Linear(896 → 256) → ReLU → Linear(256 → 1)`

A few things I added on top of the basic architecture:

- **Class weighting** in the loss function — since there are 483 hallucinated vs 206 truthful samples, without weighting the model just learns to predict "hallucinated" for everything and still gets 70% accuracy. Weighting by `n_neg/n_pos` forces it to treat both classes seriously.
- **Threshold tuning** — after training, I sweep the decision threshold on the validation set and pick the one that maximises F1. Default 0.5 is almost never optimal on imbalanced data. This was something I learned from a previous project and it made a noticeable difference here too.

### Final results

| | Accuracy | F1 | AUROC |
|---|---|---|---|
| Majority-class baseline | 70.19% | 82.49% | — |
| Train | 97.71% | 98.39% | 100% |
| Val | 73.08% | 82.50% | 66.88% |
| **Test** | **73.08%** | **81.33%** | **74.15%** |

Beats the baseline by ~4 points on accuracy and 74% AUROC shows the probe is actually learning something, not just guessing the majority class.

---

## Things I Tried That Didn't Work

**Multi-layer mean pooling:** Instead of just the last layer, I took the last 8 layers, mean-pooled token representations in each, and concatenated them into a 7168-dim vector. Intuitively more information should help — but in practice the feature space got too large for 481 training samples and performance dropped to ~60% AUROC. PCA didn't save it either.

**5-fold stratified cross-validation:** Seemed like a more rigorous evaluation approach. Problem is it reduces training samples per fold to ~468, and the smaller training set hurt the probe consistently. With only 689 total samples, a single larger split actually outperforms k-fold here.

**Deeper MLP with dropout and BatchNorm:** Tried `896 → 512 → 128 → 1` with Dropout(0.4). The train/val gap got slightly smaller but test AUROC didn't improve. Too much complexity for the dataset size.

**Logistic Regression with PCA:** Swapped the MLP for a sklearn LogisticRegression pipeline with PCA(64 components). Linear probes work well in the research literature for this kind of task, but here it underperformed the MLP — probably because the last-token features have enough non-linearity that a shallow MLP captures better.

**The pattern I noticed:** Every time I added more complexity — more layers, more features, more folds — performance got worse. The simplest version consistently won. I think that's just the reality of working with a 689-sample dataset.
