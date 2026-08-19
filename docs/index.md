# Capstone Report — Refresh / Content Opportunity Scoring

- **Author:** Sajidur Rahman Siam
- **Lane:** Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/SajidurCodes/flyrank-ml-starter
- **Date:** 2026-08-19

## 0. Abstract

Which content pages deserve a human editor's review first, and why? This project builds a leakage-checked, client-grouped model on real Google Search Console data from the FlyRank ML Internship warehouse, comparing a hand-written baseline rule, Logistic Regression, and Random Forest at predicting a >15% click decline from March to April 2026. Once checked against the label's 64.2% base rate, two of the three methods — the baseline rule and Random Forest — showed no meaningful lift over guessing, while Logistic Regression showed a genuine 15.8-point improvement in identifying the highest-priority pages. The output is a ranked, reason-coded action queue for editorial review, not an automated publishing system — a human checks every recommendation before acting.

## 1. Problem framing

**Unit of analysis:** one published content page, for one client, as of a monthly decision point (March 2026).

**Output:** a ranked priority score, a reason code (which signal drove the score — CTR-gap-vs-position, or content staleness), and an action label (`CTR_FIX`, `REFRESH`, `PROTECT`).

**The action a human takes:** a content editor reviews the top of the queue first, checking whether an underperforming page's CTR gap, or an old page's staleness, is worth an actual edit — retitling, expanding, updating facts, improving snippets.

**The cost of a wrong call:** a false positive wastes editor time reviewing a healthy page. A false negative means a genuinely declining page goes unreviewed for another month. Given the model's modest lift over base rate (see Results), this queue is explicitly a *prioritization aid*, not a high-confidence automated trigger — the cost of over-trusting it is real, and Section 5 names that directly.

**Why ML helps here at all:** a portfolio of thousands of pages can't be manually triaged by intuition alone every month. A ranked, reason-coded queue turns an unmanageable list into a short, explainable starting point — even a modest, honestly-reported lift over a hand-written rule is worth having if it's cheap to compute and clearly bounded in its claims.

## 2. Data safety

**Source:** `FlyRank/internship-warehouse` (Hugging Face, gated, read-token access).

**Tables used:** `fact_content_daily_performance` (partitioned `month=YYYY-MM`), `dim_content.parquet`.

**Date windows:** March 2026 (decision month, all features), April 2026 (label window only — never a feature). June 2026 (`_sample` table) was never touched; the dataset card flags it as the sealed natural outcome window for any past→future label.

**Deliberately excluded, and why:**
- `backlinks`, `competition`, `search_volume`, `cpc` — `dim_content` is a single unpartitioned snapshot as of the dataset's July export date, with no per-field timestamp for these columns. Only `content_updated_date` could be clamped to the decision point; the rest were excluded as an unresolved leakage risk rather than assumed safe.
- Clients without GA4 coverage as of March, and any page where `is_published` is false or `is_deleted` is true.

**Leakage risks considered:**
- The label's own input (`clicks_april`) is asserted absent from every feature column used by the model.
- `client_hash_id` and `content_hash_id` are used only for grouping and joins, never as model features.
- `content_updated_date` is clamped to `<= 2026-03-31` after an unclamped first attempt produced impossible negative "days stale" values — the snapshot-date problem was caught by noticing the numbers didn't make sense, not by schema inspection alone, and is disclosed as a partially-unresolved risk for the remaining static content fields.

**No client names, domains, URLs, or raw identifiers appear anywhere in `work/`** — only pseudonymous hashed IDs.

## 3. Baseline

A hand-written weighted rule: `score = 0.85 × normalize(ctr_gap) + 0.15 × normalize(days_stale)`.

**Why these weights, and why this is a fair comparison:** the weights are not arbitrary or tuned to the label — they come directly from signal-verification tests run *before* any model was built. CTR-gap-vs-position was tested against real month-over-month click trends and confirmed as a large, consistent effect (pages underperforming expected CTR grew +0.4 clicks/month vs. +6.1 for pages meeting/beating it). Staleness was tested the same way and came back MIXED, leaning FALSE — non-monotonic, an order of magnitude smaller effect. The baseline's weights simply reflect what was actually confirmed, which is what makes the later baseline-vs-model comparison meaningful rather than a strawman.

**Baseline numbers, same split and metric as the model:** Precision@10% = 0.667, ROC-AUC = 0.579.

## 4. Model / analysis

**Method:** Logistic Regression, chosen over Random Forest primarily for interpretability — its coefficients are directly readable, which matters more for an editorial-trust tool than a marginal accuracy gain from a black-box model. Random Forest was trained as a secondary check specifically to see whether it earned its added complexity; it did not (see Results).

**Feature list (6 total):** `ctr_march`, `avg_position_march`, `ctr_gap` (vs. a March-only position-benchmark curve), `word_count`, `char_count`, `days_stale` (clamped to the decision point).

**Deliberately left out:** `backlinks`, `competition`, `search_volume`, `cpc` — same unresolved-snapshot-leakage reasoning as Section 2.

**Target/proxy definition, one sentence:** `declined = 1` if a page's clicks fell more than 15% from March to April 2026, restricted to pages with at least 5 March clicks to avoid noisy percentage swings on near-zero-click pages.

## 5. Evaluation

**Split:** client-grouped (`GroupShuffleSplit` on `client_hash_id`), not random row-level. Directly justified by testing both: a naive random split showed 17 clients appearing in both train and test; the grouped split showed 0. Pages from the same client share unobserved structure (CMS, team quality, site-wide SEO health), so a random split risks the model learning client identity rather than transferable signal.

**Metrics, model vs. baseline, same split:**

| method | precision@10% | lift over 64.2% base rate | roc_auc |
|---|---|---|---|
| baseline rule | 0.667 | +2.5 pts | 0.579 |
| **Logistic Regression** | **0.800** | **+15.8 pts** | 0.530 |
| Random Forest | 0.667 | +2.5 pts | 0.489 |

**Why the base rate matters:** 64.2% of labeled pages are `declined`. Any random 10% slice already scores ~64% "precision" by chance. Reporting raw Precision@10% without this context would overstate two of the three results.

**What the errors look like:** false positives (model flags decline, page stays healthy) cluster on low-impression pages, where the CTR-gap calculation is noisiest — a single-digit impression swing can flip the ratio. False negatives (model misses a real decline) tend to sit near the model's own 0.5 decision boundary, consistent with weak overall discrimination (ROC-AUC ≤ 0.58 across every method tested) rather than a systematic blind spot in one feature.

## 6. Interpretation

Logistic Regression's coefficients place `ctr_gap` as the dominant driver, echoing the standalone signal test from Section 3 — this consistency across two independent tests (a simple bucket comparison, and a fitted model's coefficients) is a stronger form of evidence than either alone. `days_stale`'s coefficient is small, again consistent with the earlier MIXED/FALSE signal verdict.

**The most important negative result:** Random Forest and the baseline rule are statistically indistinguishable from guessing at the base rate. This is a legitimate, useful finding, not a failure to hide — it means added model complexity (Random Forest) bought nothing here, and a hand-written rule, while transparent, isn't meaningfully better than chance on this specific top-decile task. Only Logistic Regression showed real lift, and even then, its ROC-AUC was the *lowest* of the three — it wins narrowly at the top of the ranking, not across the whole population.

## 7. Recommendation

Real output from `work/outputs/action_playbook_queue.csv` (3,290 pages scored):

| action | reason_code | count | share |
|---|---|---|---|
| PROTECT | HEALTHY | 3,237 | 98.4% |
| CTR_FIX | CTR_GAP_ONLY | 51 | 1.5% |
| REFRESH | STALE_ONLY (low confidence) | 2 | 0.06% |

**How an editor uses this tomorrow:** start with the 51 `CTR_FIX` rows — pages ranking reasonably well but underconverting relative to position, the one signal confirmed by testing. Review the 2 `STALE_ONLY` rows with lower confidence, since staleness alone was never confirmed as predictive on its own. The 3,237 `PROTECT` rows mean "no signal fired under this threshold," not "confirmed healthy" — a distinction worth keeping in front of the editor.

**Confidence, stated explicitly:** this queue is a prioritization aid built on a 159-row validation set with ROC-AUC never exceeding 0.58. It orders pages usefully; it does not assign a trustworthy individual-page probability. Never auto-publish from this queue, and never treat a `model_score` as a calibrated per-page decline probability in client-facing reporting.

## 8. Reproducibility

**Re-run from a fresh clone:**
```bash
git clone https://github.com/SajidurCodes/flyrank-ml-starter
cd flyrank-ml-starter
pip install -r requirements.txt   
jupyter notebook work/notebooks/w03_data_contract.ipynb  
```

**Random seeds:** `random_state=42` throughout (`GroupShuffleSplit`, `train_test_split`, `RandomForestClassifier`).

**Sealed evaluation:** the w06/w07 validation numbers reported above come from a single client-grouped holdout, built and scored once in `w06_validation_audit.ipynb`; the exact split logic and the resulting metrics are committed (`w06_validation_audit.ipynb`, `work/outputs/w07_metrics.json`) so the numbers in this report are checkable against a fresh re-run, not taken on faith.

## 9. Acknowledgments & data credit

Built on the FlyRank ML Internship dataset — [flyrank.ai](https://flyrank.ai).
