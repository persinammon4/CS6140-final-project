# Status from Person 2 (personalization): week-1 ML work is done

For: Person 1 and Person 3. The AWS/cloud track is my own, so it is not covered here.
All numbers below are from my runs on the shared data and split. Test numbers come from one test run, after the settings were chosen on validation.

## 1. What is done
- **Split:** my `split_by_session` (seed 42, 70/15/15 by session) reproduces Person 1's `splits.parquet` exactly (0 mismatches over 71,102 votes, no session in two splits). Person 1's file stays the single source of truth.
- **Features:** the cleaned file had only character lengths, so I computed 6 numbers per response from the text (definitions in section 4). Saved as numbers only.
- **Global model:** logistic regression on style features, trained on both A/B orders, with the strength setting `lambda` chosen on validation.
- **Interface spec:** `interface_spec.md` v1.0 is final for the model side (9 functions, feature table, artifact files).
- **Checks passed:** my fit matches scikit-learn to 1e-8, swapping A and B gives probabilities that sum to 1, and the fitted intercept is about 0.
- **Test split:** used once, at the end.

## 2. Results (decisive votes only, meaning A or B won)
- **Data:** 71,102 English votes. Decisive votes: train 36,151, validation 7,715, test 7,739. Sessions: 43,045 / 9,224 / 9,224.
- **Baseline "always pick the longer response":** 0.5974 on validation (7,665 votes) and 0.6007 on test (7,693 votes). It reproduces Person 1's 0.6002 (51,552 votes, all splits, by characters).
- **Global model (lambda = 300):**
  - Validation: accuracy 0.6025, log loss 0.6638.
  - Test: accuracy 0.6028, log loss 0.6631.
  - Log loss of a coin flip is 0.6931, so the model knows something, but not much.
- **Model vs baseline on the baseline's votes (95% intervals, resampling whole sessions):**
  - Validation: 0.6035 vs 0.5974, difference +0.0061 [-0.0009, +0.0141].
  - Test: 0.6039 vs 0.6007, difference +0.0032 [-0.0042, +0.0112].
  - Both include 0, so there is no clear accuracy gain over "longer wins".
- **Formatting features do help the probabilities:** log loss vs a words-only model is -0.0057 [-0.0081, -0.0033], a clear difference (validation).
- **Log vs raw counts:** log loss gap +0.0017 [-0.0025, +0.0054], no clear difference. I keep log.
- **Weights (scaled units, all 6 clearly different from 0):** words +0.30, bold +0.10, headers +0.09, code +0.07, bullets +0.06, refusal -0.13. Length is the main driver.
- **Lambda:** log loss is flat from 0.01 to 300 (0.6638). Any value from 0.01 to 1,000 gives the same result. Chosen by the rule fixed in advance: lowest validation log loss.
- **Position bias:** A wins 49.5% of decisive votes.
- **Prompt overlap:** 323 of 7,665 validation votes (4.2%) share a prompt with training. Accuracy is similar on seen and unseen prompts.

## 3. What this means for the project
- **The global model is about 60%, the same as "longer wins".** Personalization has little room above that.
- **Expect a small or null personalization result.** Please do not write "personalization improves accuracy" anywhere before the Nov 8 to 12 curves exist.
- **A null result is reportable** with its mechanism. The README's Analysis section already asks why prompt type helps or does not.

## 4. Decisions that affect your work
- **Features, per response (order fixed):**
  1. `log_words`: log(1 + words), whole response.
  2. `log_headers`: log(1 + header lines), outside code blocks.
  3. `log_bullets`: log(1 + list lines), outside code blocks.
  4. `log_bold`: log(1 + `**bold**` pairs), outside code blocks.
  5. `log_code`: log(1 + number of code blocks).
  6. `refusal`: 1 if a refusal phrase is in the first 300 characters (a rule of thumb, noisy), else 0.
- **Pair feature:** `dx = features(A) - features(B)`, then divided by the per-feature scale in `scaler.json`. No mean is subtracted.
- **Model:** `P(A preferred) = sigmoid(w . dx)`. No intercept, because training on both A/B orders makes the best intercept exactly 0.
- **Confidence of a type assignment:** `1 - d1/d2` (distance to the nearest and second-nearest cluster center). It is not a probability, and values will mostly be low. Please label it "how clearly this type was picked", not "probability".
- **No sentence encoder in the cloud.** The demo uses team-written pairs with precomputed features and prompt embedding per pair.
- **Encoder for embeddings:** `all-MiniLM-L6-v2`, 384 numbers (from Person 1's files).

## 5. Files (Drive, `arena/results/`)
- `global.json`: chosen lambda, validation and test numbers, baseline, weights, scale.
- `global_model/w_global.npy` and `global_model/scaler.json`: the global model for the demo and cloud (feature order as in section 4).
- `features_raw.parquet`: 71,102 rows, 16 columns, raw counts for A and B plus id, session, split, winner. **No text.**
- `interface_spec.md`: the contract.
- `ml_week1_global_model_full.ipynb`: all week-1 code in one commented notebook.
- **Not shared:** `clean_en.parquet` holds raw prompts and responses. It stays out of the repo and out of the demo.

## 6. What each of you needs to do
**Person 1**
- Keep using `splits.parquet` as it is.
- For per-cluster fits and `user_only`: use the same 6 features, the log transform and the scale in `scaler.json`. I send the fit function on Nov 5. Your k = 0 per-cluster numbers must match my `type_only`.
- Send `clusters.parquet` and the centers on Nov 6. Fit them on the train split only, as before.
- Note: the K-Means silhouette is about 0.01, so types may add little over the global model. I will report whatever the numbers show.

**Person 3**
- Build the demo against `interface_spec.md`. Mock the functions until the real files arrive.
- The model is linear, so `p_A = sigmoid(w . dx)`, and `profile` returns the 6 named weights.
- Precomputed pair data needs `dx` and a prompt embedding per pair. I will provide the feature code (see section 7).
- Real-session check: in the test split only 897 votes remain after revealing k = 1, 289 after k = 2, and 125 after k = 3. Only k = 1 is comfortable.

## 7. Not done yet (so nobody assumes it)
- `src/features.py` and `src/split.py` exported as plain files. Not done yet.
- Simulated-user generator and the update rule (Nov 2 to 5). Not started.
- Interim model files with parity test cases (Nov 9) and the final model files (Nov 10). Not started.
- The accuracy-versus-k curve (first version Nov 8, final Nov 12). Not started.

## 8. Known limits
- Only decisive votes (72.5% of all votes) are modeled. Ties and both-bad votes are left out.
- The refusal flag is a rule of thumb.
- A few responses have extreme header counts (up to 1,244) whose cause I have not identified. The log transform limits their effect.
- Style features describe how a response looks, not whether it is right. That is my explanation for the low ceiling, and I have not tested it.
