[README (1).md](https://github.com/user-attachments/files/29887840/README.1.md)
# Balancing-Edge-Analytics-and-Data-Poisoning-Defense
This Python project implements Local Differential Privacy (LDP) for predictive maintenance in Industrial IoT. It generates synthetic sensor data, applies Laplace noise to protect privacy, trains and evaluates machine learning models, simulates attacks, benchmarks performance, creates visualizations, and validates functionality with unit tests.
# Balancing Edge Analytics and Data Poisoning Defense
### Local Differential Privacy for Predictive-Maintenance IIoT Telemetry
**File:** `combined_ldp_project.py` — single-file edition (KKKT6323, Research Topic 10)

This file merges the entire project codebase — the data simulator, the LDP
privacy engine, the predictive-maintenance model, the reconstruction-attack
threat model, the performance benchmark, the orchestration script, **and**
all pytest unit tests — into one standalone Python file. Nothing was
rewritten during the merge; only the module boundaries and internal imports
were removed.

---

## 1. What's inside `combined_ldp_project.py`

The file is organized into seven clearly marked sections, in this order:

| Section | Original module | Contents |
|---|---|---|
| 1 | `data_simulator.py` | `generate_dataset()` — synthetic IIoT vibration telemetry (RMS, kurtosis, peak frequency, crest factor, temperature) |
| 2 | `ldp_mechanism.py` | `LaplaceMechanism`, `EdgePrivacyEngine` — the edge-deployed Local Differential Privacy engine |
| 3 | `predictive_model.py` | `train_and_evaluate()` — Random Forest failure-prediction classifier |
| 4 | `threat_model.py` | `simulate_reconstruction_attack()`, `attack_sweep()` — the averaging reconstruction-attack simulation |
| 5 | `performance_eval.py` | `benchmark_pipeline()` — latency / throughput / CPU / memory overhead measurement |
| 6 | `run_experiment.py` | `main()` — orchestrates Phases I–IV end to end, writes `results/` and `figures/` |
| 7 | `tests/test_*.py` | 10 pytest unit tests (see §4 below) |

Everything lives in one shared namespace now, so no `sys.path` manipulation
or relative imports are needed — you only ever run or import this one file.

---

## 2. Deployment Instructions

### 2.1 Prerequisites

- Python 3.9+ (developed and tested on Python 3.12)
- pip

### 2.2 Install dependencies

```bash
pip install numpy pandas scikit-learn matplotlib psutil pytest --break-system-packages
```

(Drop `--break-system-packages` if you're installing into a virtual
environment, which is recommended for a permanent deployment:)

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install numpy pandas scikit-learn matplotlib psutil pytest
```

### 2.3 Place the file

Copy `combined_ldp_project.py` into any working directory you have write
access to. The script creates three subfolders **relative to wherever you
run it from**:

- `data/` — the generated synthetic telemetry CSV
- `results/` — all numeric results (CSV/JSON)
- `figures/` — all generated charts (PNG)

There is no installation step beyond having the dependencies above —
this is a single self-contained script, not a package.

### 2.4 Run the full pipeline

```bash
python3 combined_ldp_project.py
```

This executes, in order:

1. **Phase I** — generates 6,000 synthetic telemetry records (30% fault / 70% healthy) and writes them to `data/vibration_telemetry.csv`.
2. **Baseline** — trains a Random Forest on the raw (non-privatized) data. Expected accuracy: ≈ 99.7%.
3. **Phase III/IV — Equilibrium sweep** — re-trains the same model on data privatized at ε ∈ {0.1, 0.5, 1, 2, 4, 8, 16}, printing accuracy/F1 at each step. Results are written to `results/equilibrium_results.csv`.
4. **Phase II/IV — Reconstruction attack** — simulates an averaging adversary across the same ε values and N ∈ {1, 5, 20, 100} intercepted releases. Written to `results/reconstruction_attack_results.csv`.
5. **Phase IV — Performance benchmark** — measures latency/throughput/CPU for a plaintext pipeline vs. the LDP pipeline at ε ∈ {0.5, 1, 4}. Written to `results/performance_results.csv` and `results/overhead_summary.csv`.
6. **Figures** — four PNG charts are saved to `figures/`:
   - `fig1_utility_vs_epsilon.png`
   - `fig2_attack_error_vs_epsilon.png`
   - `fig3_latency_overhead.png`
   - `fig4_throughput.png`

Expected runtime: **10–20 seconds** on a typical laptop CPU (no GPU required).

**Expected console output (abridged):**
```
[Phase I] Generated 6000 telemetry samples (1800 fault / 4200 healthy)
[Baseline] accuracy=0.9967 f1=0.9944
[eps= 0.10] accuracy=0.7000 (retention=70.2%) f1=0.0088
[eps= 1.00] accuracy=0.6987 (retention=70.1%) f1=0.1308
[eps= 8.00] accuracy=0.9287 (retention=93.2%) f1=0.8780
[eps=16.00] accuracy=0.9800 (retention=98.3%) f1=0.9663
[Threat Model] Reconstruction attack sweep complete.
[Perf eps=1.0] latency_mean=0.0292 ms, throughput=34001 rec/s
All results written to results/ and figures/ directories.
```

Results are fully reproducible — all random seeds are fixed (`SEED = 42`
for data/model generation; separate fixed seeds inside the LDP engine and
the attack simulation), so re-running the script produces **identical
accuracy/F1/attack-error numbers** every time, regardless of machine.
The latency/throughput line above is a wall-clock measurement, so it will
vary with your CPU — expect the same *qualitative* pattern (LDP pipeline
slower than plaintext, but still far faster than any real IIoT sampling
rate) rather than the exact millisecond value shown.

### 2.5 Deploying the privacy engine to real edge hardware

`EdgePrivacyEngine.privatize_record()` (Section 2 of the file) depends only
on NumPy and is the one function that would actually run on field hardware.
It can be:
- Run as-is on a Raspberry Pi / Linux gateway sitting between the sensor
  bus and the MQTT broker, calling `privatize_record()` immediately before
  `mqtt_client.publish()`.
- Ported to MicroPython/CircuitPython for an ESP32-class microcontroller,
  since the mechanism itself (clip + Laplace noise) has no dependency on
  the rest of the file (dataset generation, model training, benchmarking).

The rest of the file (Sections 1, 3–6) is a simulation/evaluation harness
and is not meant to run on constrained edge hardware.

---

## 3. Configuration

A few constants near the top of Section 6 control the experiment and can
be edited directly in the file:

| Constant | Location | Default | Effect |
|---|---|---|---|
| `EPSILONS` | Section 6 | `[0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]` | Privacy budgets swept in the equilibrium test |
| `SEED` | Section 6 | `42` | Random seed for data generation and model splits |
| `FEATURE_BOUNDS` | Section 1 | per-feature `(lo, hi)` dict | Sensitivity bounds used by the Laplace mechanism |
| `RESULTS_DIR` / `FIGURES_DIR` | Section 6 | `"results"` / `"figures"` | Output folder names (relative to CWD) |

---

## 4. Test Scripts

All 10 unit tests from the original `tests/` folder are included in
Section 7 of the file and are discoverable by `pytest` directly against
this single file — no test folder or `conftest.py` needed.

### 4.1 Run the tests

```bash
pytest combined_ldp_project.py -v
```

### 4.2 Expected result

```
10 passed
```

### 4.3 What each test checks

| Test | Verifies |
|---|---|
| `test_sensitivity_and_scale` | `LaplaceMechanism` computes sensitivity (`hi - lo`) and noise scale (`sensitivity / epsilon`) correctly |
| `test_privatize_clips_before_noising` | Values outside `[lo, hi]` are clipped before noise is added |
| `test_lower_epsilon_increases_noise_variance` | A smaller ε produces higher-variance (noisier) output than a larger ε |
| `test_edge_privacy_engine_privatize_record` | `EdgePrivacyEngine.privatize_record()` returns a dict with the same keys, correctly privatized |
| `test_composed_epsilon_total` | Sequential-composition budget (`epsilon × number of features`) is computed correctly |
| `test_generate_dataset_shape_and_labels` | The synthetic dataset has the requested size and fault ratio |
| `test_generate_dataset_reproducible` | The same `random_state` always generates an identical dataset |
| `test_train_and_evaluate_baseline_accuracy_reasonable` | The Random Forest achieves >85% accuracy on raw (non-privatized) data |
| `test_more_intercepted_releases_helps_attacker` | An attacker who averages more intercepted releases (higher N) achieves lower reconstruction error |
| `test_lower_epsilon_defeats_attacker_more` | A smaller ε causes higher attacker reconstruction error than a larger ε |

### 4.4 Running only a subset

```bash
# Run only the LDP mechanism tests
pytest combined_ldp_project.py -v -k "sensitivity or privatize or epsilon or engine"

# Run only the threat-model tests
pytest combined_ldp_project.py -v -k "attacker"
```

### 4.5 Test runtime

The full test suite runs in a few seconds (well under 10s) — the slowest
tests are the ones that train a Random Forest or run 300–500 Monte Carlo
trials of the reconstruction attack on small synthetic samples.

---

## 5. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `ModuleNotFoundError: No module named 'sklearn'` (or similar) | Dependencies not installed — see §2.2 |
| `pip install` fails with an "externally managed environment" error | Add `--break-system-packages`, or use a virtual environment (§2.2) |
| Charts look different from another run | Check that `SEED` wasn't changed and that `EPSILONS` matches the default list |
| `results/` or `figures/` not where expected | These folders are created **relative to the current working directory**, not next to the script — run from the folder where you want them created |
| `pytest` reports 0 tests collected | You likely ran plain `pytest` from a directory that doesn't contain the file — always pass the filename explicitly: `pytest combined_ldp_project.py -v` |

---

## 6. File Map (quick reference)

```
combined_ldp_project.py
├── Section 1 — Data Simulator            (line   59)
├── Section 2 — LDP Mechanism             (line  125)
├── Section 3 — Predictive Model          (line  192)
├── Section 4 — Threat Model              (line  231)
├── Section 5 — Performance Evaluation    (line  268)
├── Section 6 — Main Orchestration        (line  321, main() at line 336)
├── Section 7 — Unit Tests (pytest)       (line  497)
└── Entry point: `if __name__ == "__main__":` (line 586)
```

(Search for the `# SECTION` comments in your editor to jump straight to
any part of the file.)
