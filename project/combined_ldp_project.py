"""
combined_ldp_project.py
=========================
Single-file combination of the entire Topic 10 codebase:
"Balancing Edge Analytics and Data Poisoning Defense" — Local Differential
Privacy for Predictive-Maintenance IIoT Telemetry.

This file merges, in order, the original modules:
    src/data_simulator.py      -> SECTION 1
    src/ldp_mechanism.py       -> SECTION 2
    src/predictive_model.py    -> SECTION 3
    src/threat_model.py        -> SECTION 4
    src/performance_eval.py    -> SECTION 5
    src/run_experiment.py      -> SECTION 6 (main orchestration)
    tests/test_*.py            -> SECTION 7 (pytest unit tests)

No functionality, class, or function was changed during the merge — only
the module boundaries were removed and internal `from x import y` statements
were deleted because everything now lives in one namespace.

HOW TO RUN
----------
1) Run the full Phase I-IV experiment (generates data, trains models, runs
   the threat simulation, benchmarks performance, writes results/ and
   figures/ to the current working directory):

       python3 combined_ldp_project.py

2) Run the unit tests instead (requires pytest):

       pytest combined_ldp_project.py -v

   (pytest auto-discovers the test_* functions in Section 7 regardless of
   which file they live in.)
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

import psutil


# =====================================================================
# SECTION 1 — data_simulator.py
# =====================================================================
"""
Simulates a fleet of industrial machines (e.g., rotating equipment / motor
bearings) instrumented with vibration IoT sensors reporting telemetry used
for predictive maintenance.

Each sample represents one telemetry window (e.g., a 1-second accelerometer
window already reduced to standard condition-monitoring features by the edge
device). Features are modelled on well-known vibration-analysis descriptors:

    rms          - Root Mean Square amplitude (overall energy)
    kurtosis     - Impulsiveness of the signal (spikes -> bearing defects)
    peak_freq    - Dominant vibration frequency (Hz)
    crest_factor - Peak / RMS ratio (early fault indicator)
    temperature  - Bearing/housing temperature (deg C)

Label: 1 = machine will fail within the prediction horizon, 0 = healthy.
"""


def generate_dataset(n_samples: int = 6000, fault_ratio: float = 0.30,
                      random_state: int = 42) -> pd.DataFrame:
    """Generate a synthetic predictive-maintenance vibration dataset."""
    rng = np.random.default_rng(random_state)
    n_fault = int(n_samples * fault_ratio)
    n_healthy = n_samples - n_fault

    # --- Healthy population ---
    healthy = pd.DataFrame({
        "rms":          rng.normal(0.38, 0.11, n_healthy).clip(0.05, None),
        "kurtosis":     rng.normal(3.3, 0.9, n_healthy).clip(1.5, None),
        "peak_freq":    rng.normal(55.0, 12.0, n_healthy).clip(10, None),
        "crest_factor": rng.normal(3.4, 0.6, n_healthy).clip(1.0, None),
        "temperature":  rng.normal(46.0, 6.0, n_healthy).clip(10, None),
    })
    healthy["label"] = 0

    # --- Degrading / pre-failure population ---
    fault = pd.DataFrame({
        "rms":          rng.normal(0.68, 0.16, n_fault).clip(0.05, None),
        "kurtosis":     rng.normal(6.0, 1.6, n_fault).clip(1.5, None),
        "peak_freq":    rng.normal(110.0, 22.0, n_fault).clip(10, None),
        "crest_factor": rng.normal(5.3, 0.9, n_fault).clip(1.0, None),
        "temperature":  rng.normal(64.0, 7.0, n_fault).clip(10, None),
    })
    fault["label"] = 1

    df = pd.concat([healthy, fault], ignore_index=True)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df


FEATURE_COLUMNS = ["rms", "kurtosis", "peak_freq", "crest_factor", "temperature"]

# Physically-informed clipping bounds used for DP sensitivity calculation.
FEATURE_BOUNDS = {
    "rms":          (0.0, 1.5),
    "kurtosis":     (0.0, 12.0),
    "peak_freq":    (0.0, 200.0),
    "crest_factor": (0.0, 9.0),
    "temperature":  (0.0, 100.0),
}


# =====================================================================
# SECTION 2 — ldp_mechanism.py
# =====================================================================
"""
Edge-side Local Differential Privacy (LDP) engine.

Implements the classical Laplace Mechanism (Dwork & Roth, 2014) applied
independently, per-feature, at the IoT edge node BEFORE telemetry ever
leaves the machine.
"""


@dataclass
class LaplaceMechanism:
    """Local Differential Privacy Laplace mechanism for a bounded feature."""
    lo: float
    hi: float
    epsilon: float

    @property
    def sensitivity(self) -> float:
        return self.hi - self.lo

    @property
    def scale(self) -> float:
        eps = max(self.epsilon, 1e-6)
        return self.sensitivity / eps

    def privatize(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        x_clipped = np.clip(x, self.lo, self.hi)
        noise = rng.laplace(loc=0.0, scale=self.scale, size=x_clipped.shape)
        return x_clipped + noise


class EdgePrivacyEngine:
    """Applies per-feature Laplace LDP across a whole telemetry record set."""

    def __init__(self, feature_bounds: dict, epsilon: float, seed: int = 7):
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.mechanisms = {
            feat: LaplaceMechanism(lo, hi, epsilon)
            for feat, (lo, hi) in feature_bounds.items()
        }

    def privatize_dataframe(self, df, feature_columns):
        """Return a new DataFrame with privatized feature columns."""
        out = df.copy()
        for feat in feature_columns:
            out[feat] = self.mechanisms[feat].privatize(df[feat].values, self.rng)
        return out

    def privatize_record(self, record: dict) -> dict:
        """Privatize a single telemetry record (dict of feature -> value)."""
        return {
            feat: float(self.mechanisms[feat].privatize(
                np.array([val]), self.rng)[0])
            for feat, val in record.items()
        }

    @property
    def composed_epsilon_total(self) -> float:
        """Conservative sequential-composition privacy budget across all
        features released per telemetry window."""
        return self.epsilon * len(self.mechanisms)


# =====================================================================
# SECTION 3 — predictive_model.py
# =====================================================================
"""
Central-server side predictive maintenance model.

Trains a classifier that predicts imminent machine failure from telemetry
features.
"""


def train_and_evaluate(df, feature_columns, label_column="label",
                        test_size=0.25, random_state=42):
    """Train a RandomForest failure classifier and return test-set metrics."""
    X = df[feature_columns].values
    y = df[label_column].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=150, max_depth=8, random_state=random_state, n_jobs=-1
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }


# =====================================================================
# SECTION 4 — threat_model.py
# =====================================================================
"""
Confidentiality threat simulation: a curious/compromised central server (or
an eavesdropper who has captured the MQTT/HTTPS telemetry stream) attempts
to reconstruct the true proprietary sensor readings of a target machine.
"""


def simulate_reconstruction_attack(true_value: float, sensitivity: float,
                                    epsilon: float, n_intercepted: int,
                                    n_trials: int = 500, seed: int = 11):
    """Simulate an averaging reconstruction attack."""
    rng = np.random.default_rng(seed)
    scale = sensitivity / max(epsilon, 1e-6)

    errors = []
    for _ in range(n_trials):
        noisy_releases = true_value + rng.laplace(0.0, scale, size=n_intercepted)
        estimate = noisy_releases.mean()
        errors.append(abs(estimate - true_value))

    errors = np.array(errors)
    return {"mae": float(errors.mean()), "std": float(errors.std())}


def attack_sweep(true_value, sensitivity, epsilons, intercept_counts):
    """Run the reconstruction attack across a grid of (epsilon, N) values."""
    rows = []
    for eps in epsilons:
        for n in intercept_counts:
            res = simulate_reconstruction_attack(true_value, sensitivity, eps, n)
            rows.append({"epsilon": eps, "n_intercepted": n, **res})
    return rows


# =====================================================================
# SECTION 5 — performance_eval.py
# =====================================================================
"""
Empirical "Security-Privacy Equilibrium" test (Phase IV).

Measures the computational cost the Local Differential Privacy engine adds
to the edge-transmission pipeline, benchmarked against a plaintext
(no-privacy) baseline.
"""


def _process_metrics():
    p = psutil.Process(os.getpid())
    return p.cpu_times(), p.memory_info().rss


def benchmark_pipeline(records, process_fn, warmup=50):
    """Benchmark `process_fn` applied to each record in `records`."""
    proc = psutil.Process(os.getpid())

    # Warm-up
    for r in records[:warmup]:
        process_fn(r)

    mem_before = proc.memory_info().rss
    cpu_before = proc.cpu_times()
    wall_start = time.perf_counter()

    latencies = []
    for r in records:
        t0 = time.perf_counter()
        process_fn(r)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    wall_end = time.perf_counter()
    cpu_after = proc.cpu_times()
    mem_after = proc.memory_info().rss

    total_wall = wall_end - wall_start
    total_cpu = (cpu_after.user - cpu_before.user) + (cpu_after.system - cpu_before.system)

    latencies = np.array(latencies)
    return {
        "latency_ms_mean": float(latencies.mean()),
        "latency_ms_p95": float(np.percentile(latencies, 95)),
        "throughput_rps": float(len(records) / total_wall),
        "cpu_percent": float(100.0 * total_cpu / total_wall) if total_wall > 0 else 0.0,
        "memory_delta_kb": float((mem_after - mem_before) / 1024.0),
    }


# =====================================================================
# SECTION 6 — run_experiment.py (main orchestration)
# =====================================================================
"""
End-to-end experiment runner for:
    "Balancing Edge Analytics and Data Poisoning Defense" (Topic 10)
    Local Differential Privacy for Predictive-Maintenance IIoT Telemetry
"""

RESULTS_DIR = "results"
FIGURES_DIR = "figures"

EPSILONS = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
SEED = 42


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    print("=" * 70)
    print("Phase I-IV Experiment: LDP for Predictive Maintenance IIoT")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Phase I data generation (system telemetry)
    # ---------------------------------------------------------------
    df = generate_dataset(n_samples=6000, fault_ratio=0.30, random_state=SEED)
    df.to_csv(os.path.join("data", "vibration_telemetry.csv"), index=False)
    print(f"[Phase I] Generated {len(df)} telemetry samples "
          f"({df['label'].sum()} fault / {len(df) - df['label'].sum()} healthy)")

    # ---------------------------------------------------------------
    # Baseline model (no privacy) - upper bound on utility
    # ---------------------------------------------------------------
    baseline_metrics = train_and_evaluate(df, FEATURE_COLUMNS)
    print(f"[Baseline] accuracy={baseline_metrics['accuracy']:.4f} "
          f"f1={baseline_metrics['f1']:.4f}")

    # ---------------------------------------------------------------
    # Phase III/IV: Security-Privacy Equilibrium sweep across epsilon
    # ---------------------------------------------------------------
    equilibrium_rows = []
    for eps in EPSILONS:
        engine = EdgePrivacyEngine(FEATURE_BOUNDS, epsilon=eps, seed=SEED)
        priv_df = engine.privatize_dataframe(df, FEATURE_COLUMNS)
        metrics = train_and_evaluate(priv_df, FEATURE_COLUMNS)
        row = {
            "epsilon": eps,
            "epsilon_total_composed": engine.composed_epsilon_total,
            **metrics,
            "accuracy_retention_pct": 100.0 * metrics["accuracy"] / baseline_metrics["accuracy"],
        }
        equilibrium_rows.append(row)
        print(f"[eps={eps:>5.2f}] accuracy={metrics['accuracy']:.4f} "
              f"(retention={row['accuracy_retention_pct']:.1f}%) f1={metrics['f1']:.4f}")

    equilibrium_df = pd.DataFrame(equilibrium_rows)
    equilibrium_df.to_csv(os.path.join(RESULTS_DIR, "equilibrium_results.csv"), index=False)

    # ---------------------------------------------------------------
    # Phase II/IV: Reconstruction-attack (privacy strength validation)
    # ---------------------------------------------------------------
    target_true_value = float(df.loc[df.label == 1, "rms"].mean())
    sensitivity = FEATURE_BOUNDS["rms"][1] - FEATURE_BOUNDS["rms"][0]
    attack_rows = attack_sweep(
        true_value=target_true_value,
        sensitivity=sensitivity,
        epsilons=EPSILONS,
        intercept_counts=[1, 5, 20, 100],
    )
    attack_df = pd.DataFrame(attack_rows)
    attack_df.to_csv(os.path.join(RESULTS_DIR, "reconstruction_attack_results.csv"), index=False)
    print("[Threat Model] Reconstruction attack sweep complete.")

    # ---------------------------------------------------------------
    # Phase IV: Performance overhead benchmarking
    # ---------------------------------------------------------------
    import json as _json
    records = df[FEATURE_COLUMNS].to_dict(orient="records")

    def plaintext_fn(r):
        return _json.loads(_json.dumps(r))

    perf_rows = []
    plaintext_perf = benchmark_pipeline(records, plaintext_fn)
    perf_rows.append({"epsilon": "plaintext (no DP)", **plaintext_perf})

    for eps in [0.5, 1.0, 4.0]:
        engine = EdgePrivacyEngine(FEATURE_BOUNDS, epsilon=eps, seed=SEED)

        def ldp_and_serialize(r, engine=engine):
            return _json.loads(_json.dumps(engine.privatize_record(r)))

        perf = benchmark_pipeline(records, ldp_and_serialize)
        perf_rows.append({"epsilon": eps, **perf})
        print(f"[Perf eps={eps}] latency_mean={perf['latency_ms_mean']:.4f} ms, "
              f"throughput={perf['throughput_rps']:.0f} rec/s")

    perf_df = pd.DataFrame(perf_rows)
    perf_df.to_csv(os.path.join(RESULTS_DIR, "performance_results.csv"), index=False)

    baseline_latency = plaintext_perf["latency_ms_mean"]
    baseline_throughput = plaintext_perf["throughput_rps"]
    overhead_rows = []
    for row in perf_rows[1:]:
        overhead_rows.append({
            "epsilon": row["epsilon"],
            "latency_overhead_pct": 100.0 * (row["latency_ms_mean"] - baseline_latency) / baseline_latency,
            "throughput_change_pct": 100.0 * (row["throughput_rps"] - baseline_throughput) / baseline_throughput,
        })
    overhead_df = pd.DataFrame(overhead_rows)
    overhead_df.to_csv(os.path.join(RESULTS_DIR, "overhead_summary.csv"), index=False)

    with open(os.path.join(RESULTS_DIR, "baseline_metrics.json"), "w") as f:
        json.dump(baseline_metrics, f, indent=2)

    # ---------------------------------------------------------------
    # Figures
    # ---------------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(equilibrium_df["epsilon"], equilibrium_df["accuracy"] * 100,
              marker="o", color="#2563eb", label="Model Accuracy")
    ax1.axhline(baseline_metrics["accuracy"] * 100, color="gray", linestyle="--",
                label="Baseline (no privacy)")
    ax1.axhline(90, color="#dc2626", linestyle=":", label="90% Utility Threshold")
    ax1.set_xscale("log")
    ax1.set_xlabel("Privacy Budget (epsilon, log scale) — lower = stronger privacy")
    ax1.set_ylabel("Failure-Prediction Accuracy (%)")
    ax1.set_title("Security-Privacy Equilibrium: Utility vs. Privacy Budget")
    ax1.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig1_utility_vs_epsilon.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n in sorted(attack_df["n_intercepted"].unique()):
        sub = attack_df[attack_df["n_intercepted"] == n]
        ax.plot(sub["epsilon"], sub["mae"], marker="s", label=f"N={n} intercepted releases")
    ax.set_xscale("log")
    ax.set_xlabel("Privacy Budget (epsilon, log scale)")
    ax.set_ylabel("Attacker Mean Absolute Reconstruction Error")
    ax.set_title("Confidentiality: Reconstruction-Attack Error vs. Privacy Budget")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig2_attack_error_vs_epsilon.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = [str(r["epsilon"]) for r in perf_rows]
    latencies = [r["latency_ms_mean"] for r in perf_rows]
    ax.bar(labels, latencies, color=["#6b7280"] + ["#2563eb"] * (len(labels) - 1))
    ax.set_xlabel("Configuration (epsilon)")
    ax.set_ylabel("Mean per-record Latency (ms)")
    ax.set_title("Edge Processing Latency: Plaintext vs. LDP-Privatized Pipeline")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig3_latency_overhead.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    throughputs = [r["throughput_rps"] for r in perf_rows]
    ax.bar(labels, throughputs, color=["#6b7280"] + ["#16a34a"] * (len(labels) - 1))
    ax.set_xlabel("Configuration (epsilon)")
    ax.set_ylabel("Throughput (records/sec)")
    ax.set_title("Edge Processing Throughput: Plaintext vs. LDP-Privatized Pipeline")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig4_throughput.png"), dpi=150)
    plt.close(fig)

    print("\nAll results written to results/ and figures/ directories.")
    print("=" * 70)


# =====================================================================
# SECTION 7 — pytest unit tests (from tests/test_*.py)
# =====================================================================

# ---- from tests/test_ldp_mechanism.py ----

def test_sensitivity_and_scale():
    mech = LaplaceMechanism(lo=0.0, hi=1.5, epsilon=1.0)
    assert mech.sensitivity == 1.5
    assert mech.scale == 1.5


def test_privatize_clips_before_noising():
    rng = np.random.default_rng(0)
    mech = LaplaceMechanism(lo=0.0, hi=1.0, epsilon=1000.0)  # near-zero noise
    out = mech.privatize(np.array([5.0, -5.0, 0.5]), rng)
    assert out[0] < 1.5
    assert out[1] > -1.5


def test_lower_epsilon_increases_noise_variance():
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(1)
    mech_strong_privacy = LaplaceMechanism(lo=0.0, hi=1.0, epsilon=0.1)
    mech_weak_privacy = LaplaceMechanism(lo=0.0, hi=1.0, epsilon=10.0)

    out_strong = mech_strong_privacy.privatize(np.full(2000, 0.5), rng1)
    out_weak = mech_weak_privacy.privatize(np.full(2000, 0.5), rng2)

    assert np.std(out_strong) > np.std(out_weak)


def test_edge_privacy_engine_privatize_record():
    bounds = {"rms": (0.0, 1.5), "temperature": (0.0, 100.0)}
    engine = EdgePrivacyEngine(bounds, epsilon=1.0, seed=42)
    record = {"rms": 0.4, "temperature": 45.0}
    out = engine.privatize_record(record)
    assert set(out.keys()) == set(record.keys())
    assert isinstance(out["rms"], float)


def test_composed_epsilon_total():
    bounds = {"a": (0, 1), "b": (0, 1), "c": (0, 1)}
    engine = EdgePrivacyEngine(bounds, epsilon=0.5, seed=1)
    assert engine.composed_epsilon_total == 1.5


# ---- from tests/test_predictive_model.py ----

def test_generate_dataset_shape_and_labels():
    df = generate_dataset(n_samples=500, fault_ratio=0.3, random_state=1)
    assert len(df) == 500
    assert set(df["label"].unique()) == {0, 1}
    assert abs(df["label"].mean() - 0.3) < 0.05


def test_generate_dataset_reproducible():
    df1 = generate_dataset(n_samples=200, random_state=5)
    df2 = generate_dataset(n_samples=200, random_state=5)
    assert df1.equals(df2)


def test_train_and_evaluate_baseline_accuracy_reasonable():
    df = generate_dataset(n_samples=1500, random_state=42)
    metrics = train_and_evaluate(df, FEATURE_COLUMNS)
    assert metrics["accuracy"] > 0.85
    assert 0 <= metrics["f1"] <= 1


# ---- from tests/test_threat_model.py ----

def test_more_intercepted_releases_helps_attacker():
    res_few = simulate_reconstruction_attack(
        true_value=0.5, sensitivity=1.5, epsilon=1.0, n_intercepted=1, n_trials=300)
    res_many = simulate_reconstruction_attack(
        true_value=0.5, sensitivity=1.5, epsilon=1.0, n_intercepted=200, n_trials=300)
    assert res_many["mae"] < res_few["mae"]


def test_lower_epsilon_defeats_attacker_more():
    res_strong_privacy = simulate_reconstruction_attack(
        true_value=0.5, sensitivity=1.5, epsilon=0.1, n_intercepted=10, n_trials=300)
    res_weak_privacy = simulate_reconstruction_attack(
        true_value=0.5, sensitivity=1.5, epsilon=10.0, n_intercepted=10, n_trials=300)
    assert res_strong_privacy["mae"] > res_weak_privacy["mae"]


# =====================================================================
# ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    main()
