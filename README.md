# Train Condition Monitoring — RailGuard

NebulaX Train Condition Monitoring hackathon submission app.

## Current scope

Rail Corrugation only: feature extraction, model training/benchmarking, and a Streamlit app (single-file diagnosis + batch triage producing `rail_predictions.csv`).

## Designed for the end user, not the model

The classifier answers *which class*. An operator making a split-second call needs *how bad, how urgent, what do I do, and why should I believe it*. The app is built around that gap:

- **Action first.** The primary output is a recommended action and a deadline ("Raise a maintenance order and inspect the Side I rail — within 72 hours"), not a class label and a probability.
- **Severity in context.** A reading is graded against the spread of *healthy* training recordings, so the app can say "higher than 99% of healthy recordings" instead of showing a bare number with no reference point.
- **Plain-English reasoning.** "Side I is vibrating 6.8× harder than Side II — the signature of corrugation on that rail", rather than `side_rms_ratio: 6.8`.
- **Physical orientation.** A train schematic highlights the affected rail and axle positions, because operators think in layout, not channel numbers.
- **Reliability in words.** "High confidence" / "Borderline — verify manually before acting", instead of `93.5%`.
- **Worst-first triage.** Batch results are ordered by severity, with a count of how many of the batch actually need attention.
- **Operator / Engineer toggle.** The operator view stays uncluttered; the engineer view adds waveforms, spectra, class probabilities, raw indicators and cross-validation detail. Feature names are translated too — `ch74_dominant_freq` becomes "Car 5, position 5 shock — the frequency the rail is ringing at".

Severity bands and maintenance timeframes live in one place (`src/rail_interpretation.py`) so a real operator can tune them to their own standards. They are an operating policy for this prototype, not certified engineering thresholds, and the app says so.

## Rail Corrugation — schema (verified against `Rail_Corrugation_Info_Kit.md`)

Each recording is a 129-column CSV, 10,000 rows (1 second at 10 kHz):

- Column 1: `Rotating speed` — a toothed-wheel sensor (90 teeth, 0.85 m wheel) that toggles 0/1 as teeth pass.
- Columns 2–129: 8 cars × 8 axle positions × (vibration, shock), e.g. `Vibration of bearing in position 1 of car 1`, `Shock of bearing in position 1 of car 1`, ...
- Axle positions 1, 3, 5, 7 = Side I; positions 2, 4, 6, 8 = Side II.
- Labels: `Normal`, `Side I`, `Side II` (Normal/Side I/Side II mean each side's own condition — a file can have one side faulty and the other normal).

## Run the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Without a trained model at `models/rail_model.joblib`, the Rail Corrugation tabs will explain that training hasn't run yet rather than failing silently.

## Train the model

```bash
python -m training.train_rail --train-dir path/to/Train --labels path/to/Train_Labels.csv
```

This benchmarks Random Forest, Gradient Boosting, and Logistic Regression under identical stratified 5-fold cross-validation (scored on macro F1, the subsystem's actual grading metric), keeps the best-scoring model, and writes:

- `models/rail_model.joblib` — the trained model.
- `models/rail_model_meta.json` — CV scores per candidate model, class distribution, top feature importances, and `healthy_baselines` (percentile spread of key indicators across the Normal files). The app reads this file directly, so reported numbers always reflect what you actually trained, not a hardcoded claim.

`healthy_baselines` is what powers severity grading. A model trained before that field existed still works — the app just says severity grading is unavailable and asks you to re-run training, rather than inventing a number.

**Split assumption** (stated per the challenge's "justify your split" guidance): the Info Kit documents no grouping column (e.g. run ID or operating condition) alongside each file, and every file is already one self-contained 1-second recording, so a stratified k-fold over file labels is used. If a future data drop adds a grouping variable, switch to `GroupKFold` instead to keep recordings from the same run out of both sides of a split.

## Generate a submission

```bash
python predict.py --input path/to/Test --output rail_predictions.csv
```

Produces the exact `file_id,prediction` schema required for submission, from every CSV file in the input directory.

## Feature engineering

- **Per-channel** (all 128 raw channels): RMS, std, peak, peak-to-peak, kurtosis, skew, dominant frequency, crest factor, spectral centroid.
- **Per-side aggregate** (mean vibration-only waveform for Side I / Side II, matching the app's plotted signal): the same stats plus frequency-band energy ratios (0–500 Hz, 500–1500 Hz, 1500 Hz–Nyquist).
- **Side-vs-side asymmetry**: RMS/std/peak/crest-factor ratios and dominant-frequency difference — the direct Side I vs. Side II localisation signal.
- **Speed-normalised wavelength**: corrugation wavelength is roughly speed-invariant while raw vibration frequency isn't, so `speed / dominant_frequency` is included per side as a more physically stable indicator than frequency alone. Speed itself is derived by counting 0/1 transitions of the toothed-wheel signal when that column looks binary (falls back to a plain mean/std summary otherwise, in case a future data drop pre-computes speed instead).

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Tests use synthetic recordings matching the real 129-column schema (see `tests/conftest.py`) — the actual competition dataset is never committed to this repo, per the submission rules.
