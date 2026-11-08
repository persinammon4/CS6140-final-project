# Interface spec (v1.0, ready to send)

Owner: Ibrahim (Person 2). Readers: Person 3 (demo) and the Cloud track.
Status: v1.0. All decisions are confirmed: log transform, confidence definition, D1 (scale only, no intercept), D2 (no encoder in the cloud). The global model is trained and its files follow the feature order in section 1 (see `results/global.json`, `results/global_model/w_global.npy`, `results/global_model/scaler.json`). Only D5 (update strength for personal updates) is open, and it does not change any function shape.

## 0. Ground rules
- Code lives in `src/features.py` and `src/personalize.py`.
- Pure Python and NumPy only. No scikit-learn, PyTorch or pandas in these functions, so they run unchanged in AWS Lambda.
- No randomness, no network, no file reads except `load_artifacts`. Same inputs give the same outputs.
- `d` = number of features = 6. `K` = number of prompt types (clusters). `e` = embedding size = 384 (`all-MiniLM-L6-v2`).
- Label `y` = 1 if response A was preferred, 0 if B. Soft labels (0.5 for a tie) are allowed in fitting only. Ties and both-bad votes are never used at prediction time.

## 1. Features (log transform confirmed Nov 1; log vs raw showed no clear difference, so log is kept)
Feature order is fixed. It is also stored in `scaler.json` as `feature_names`.

| # | Name | Definition on one response |
|---|---|---|
| 0 | `log_words` | `log(1 + words)`, words = `len(text.split())` |
| 1 | `log_headers` | `log(1 + n)`, n = lines starting with 1 to 6 `#` then a space, outside code blocks |
| 2 | `log_bullets` | `log(1 + n)`, n = lines starting with `-`, `*`, `+`, `•`, or a number then `.` or `)`, outside code blocks |
| 3 | `log_bold` | `log(1 + n)`, n = count of `**text**` pairs, outside code blocks |
| 4 | `log_code` | `log(1 + n)`, n = (count of triple-backtick fences) // 2 |
| 5 | `refusal` | 1 if a refusal phrase appears in the first 300 characters, else 0 (rule of thumb, noisy) |

Code blocks: text between triple-backtick fences (and an unclosed trailing block) is removed before counting headers, bullets and bold, because code comments like `# note` are not headers. Words, the code-block count and the refusal flag use the whole response.

Why the log: the raw counts have heavy tails (for example words up to 27,418 and headers up to 1,244 in the data). Max divided by the 99th percentile is 10 to 62 times on the raw scale and 1.3 to 2.3 times after the log (measured on the English data), so a few votes no longer dominate the fit.

## 2. Functions

### 2.1 `response_features(text) -> ndarray (d,)`  (features.py, offline use only)
- In: response text (str). Out: raw feature vector of one response, in the order above.
- Not called in Lambda in the default design (see D2). The demo uses precomputed pair data.

### 2.2 `pair_features(a_raw, b_raw) -> ndarray (d,)`
- In: raw feature vectors of response A and response B, each shape `(d,)`.
- Out: `dx_raw = a_raw - b_raw`, shape `(d,)`.

### 2.3 `scale_dx(dx_raw, scaler) -> ndarray (d,)`
- In: `dx_raw` shape `(d,)`; `scaler` dict with `feature_names` (list of d str) and `scale` (list of d positive floats).
- Out: `dx = dx_raw / scale`. No mean is subtracted (decision D1, confirmed Nov 1: scale only, no intercept).

### 2.4 `load_artifacts(path) -> dict`
- In: folder path (for example `artifacts/model_v1/`).
- Out: dict with keys:
  - `w_global`: ndarray `(d,)`
  - `w_clusters`: ndarray `(K, d)`
  - `centers`: ndarray `(K, e)`
  - `scaler`: dict (`feature_names`, `scale`)
  - `config`: dict (`model_version`, `encoder_name`, `K`, `lam_update`, `split_seed`, `date`, `git_hash`, `dataset_id`, optional `type_labels` mapping type id (as str) to label)
- Raises `FileNotFoundError` or `ValueError` if a file is missing or shapes disagree.

### 2.5 `assign_type(artifacts, prompt_embs) -> (int, float)`
- In: `prompt_embs` ndarray `(n, e)` with n >= 1, the prompt embeddings seen so far (one per revealed vote).
- Out: `(type_id, confidence)`.
  - `type_id`: index of the cluster center nearest (Euclidean distance) to the mean of `prompt_embs`.
  - `confidence` (decided Nov 1): `1 - d1/d2`, where d1 and d2 are the distances to the nearest and second-nearest centers; 0 if d2 = 0. Range 0 to 1. Expect low values, because the clusters are weakly separated (silhouette about 0.01).
- With K = 1 it returns `(0, 0.0)`.

### 2.6 `init_user(artifacts, type_id) -> dict`
- In: type id (int, 0 to K-1).
- Out: `user = {"type_id": int, "w": ndarray (d,) equal to w_clusters[type_id], "n_votes": 0}`.

### 2.7 `update_user(artifacts, user, dx_list, y_list) -> dict`
- In: `user` (from `init_user`); `dx_list` list or array of standardized differences, shape `(n, d)`; `y_list` list of n labels in [0, 1] (1 if A preferred).
- Out: new `user` dict with the same `type_id`, new `w`, `n_votes = n`.
- Fits `minimize sum logloss + lam * ||w - w_prior||^2` with `w_prior = w_clusters[user["type_id"]]` and `lam = config["lam_update"]`.
- Rules:
  - **Recompute from scratch.** The result depends only on the prior and the full vote list passed in. It never uses `user["w"]`. Same input gives the same output (idempotent). The Cloud track relies on this.
  - **n = 0 returns the prior unchanged.**
  - If the type changes later (more prompts seen), the caller calls `init_user` with the new type, then `update_user` with all votes.
- Raises `ValueError` if `len(dx_list) != len(y_list)` or a row is not length d.

### 2.8 `predict(user, dx) -> float`
- In: `user`; `dx` standardized difference, shape `(d,)`.
- Out: `p_A = sigmoid(w . dx)`, probability that response A is preferred, in (0, 1).
- Symmetry: `predict(user, -dx) == 1 - predict(user, dx)`.

### 2.9 `profile(artifacts, user) -> dict`
- Out:
  - `type_id`: int
  - `type_label`: str (from `config["type_labels"]` if present, else `"Type <id>"`)
  - `weights`: dict feature name to weight (float)
  - `delta_vs_global`: dict feature name to (`user w` minus `w_global`)
  - `n_votes`: int

## 3. Example call sequence (checked with a throwaway mock on random numbers)
```python
art = load_artifacts("artifacts/model_v1/")
dx = scale_dx(pair_features(a_raw, b_raw), art["scaler"])      # per pair, precomputed in the demo
type_id, conf = assign_type(art, prompt_embs_so_far)            # (n, 384)
user = init_user(art, type_id)
user = update_user(art, user, dx_list_so_far, y_list_so_far)    # all votes so far, every time
p_A_next = predict(user, dx_next)
info = profile(art, user)
```

## 4. Artifact files (`artifacts/model_v1/`)
| File | Content |
|---|---|
| `w_global.npy` | `(d,)` |
| `w_clusters.npy` | `(K, d)` |
| `centers.npy` | `(K, e)` |
| `scaler.json` | `feature_names`, `scale` (per-feature root mean square of `dx_raw` over the symmetrized train data) |
| `config.json` | see 2.4 |

Interim version (global only, Nov 9, PROVISIONAL): `K = 1`, `w_clusters = w_global[None, :]`, `centers` = one row (the mean train prompt embedding).

## 5. Decisions and open items
| Item | Status | Note |
|---|---|---|
| Log transform of count features | Decided yes (Nov 1) | Checked on validation: log vs raw log-loss gap +0.0017, 95% interval [-0.0025, +0.0054], no clear difference. Log is kept. Shapes in this spec do not depend on it |
| D1 scaling and intercept | Decided: scale only, no intercept (Nov 1) | With training on both A/B orders the best intercept is exactly 0. Checked on the real data: fitted intercept about 6.5e-19 |
| D2 encoder in Lambda | Decided: no encoder in the cloud (Nov 1). Precomputed `dx` and embeddings per demo pair | An `encode_prompt` function can be added later without changing any function above |
| Confidence definition | Decided: `1 - d1/d2` (Nov 1) | Not a probability. Label it "how clearly this type was picked" in the demo |
| `lam_update` (D5) | Open. Provisional: chosen on validation from {0.1, 1, 10, 100} | Asked when tuning the update rule (Nov 4 to 5) |

## 6. Change rule
Once this spec is sent to Person 3 and the cloud track, a signature change needs all three people to agree.
