# StormCost — Solution Write-up

## Summary
Two-stage gradient-boosting model on structured fields plus frozen contextual narrative embeddings,
validated leave-years-out, with metric-aware post-processing (decision threshold + dollar scale factors).

**Honest nested CV score:** 48.13  (occurrence 0.797, magnitude 0.155).
Reference points: all-zeros = 1, constant mean ~ 2, perfect = 100.

## Pipeline
1. **Text:** `sentence-transformers/all-MiniLM-L6-v2` (frozen, no fine-tuning). Event narratives -> 64 PCA dims,
   episode narratives -> 16 PCA dims. Unique texts are encoded once. No TF-IDF / bag-of-words /
   n-gram counters / averaged static embeddings are used.
2. **Structured features:** year, month, lon/lat, injuries/deaths (blank -> 0), magnitude (blank kept as NaN),
   magnitude type, tornado fields, event type, state; narrative lengths, episode-missing flag, redaction counts;
   episode group size and injury/death totals (label-free, computed within each file separately).
3. **Models:** LightGBM classifier for P(damage > 0) and LightGBM regressor for log1p(damage) trained on damaged rows only.
   Test predictions are the average of the 5 fold models x 1 seed(s).
4. **Decision rule:** if P(damage) >= 0.35 predict expm1(log-pred) x k, otherwise exactly 0.
   Global k = 5; per-event-type k: Flash Flood: 8, Flood: 12, Hail: 12, Strong Wind: 3, Winter Storm: 3, Ice Storm: 12, Winter Weather: 3, Wildfire: 8, Tropical Storm: 12, Heavy Rain: 12, Heavy Snow: 8.
5. **Validation:** GroupKFold over calendar years (5 folds) to mimic the disjoint-year test set;
   post-processing settings are evaluated with nested validation to avoid optimistic bias.

## Results (out-of-fold, leave-years-out)
- OOF AUC for occurrence: 0.963
- Honest CV score: 48.13
- Total runtime: 24.2 min on CUDA (budget 90 min)

## Rule compliance
Contextual encoder only; `episode_group` not used as a feature; test placeholder never used; public leaderboard never used
for tuning; predictions are finite, non-negative and exactly 0 for "no damage"; no external data or database lookup.

## Reproducibility
Seeds are fixed (seed=42); all settings live in the `CFG` dictionary; embeddings are cached.
GPU encoding can differ at the last decimal places between runs.

## Limitations & ideas
- Early stopping uses the validation fold (slight optimism).
- Dollar tail is driven by a handful of events; per-type scale factors can overfit rare types.
- Ideas: weight the regressor by damage size, quantile/Tweedie objectives, larger encoder, fine-tuning,
  era-specific models, more seeds.
