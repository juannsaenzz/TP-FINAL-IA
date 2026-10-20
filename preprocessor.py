"""
preprocessor.py — Data Preprocessing for ART1 Network

Converts the RIASEC career prediction dataset into binary vectors
suitable for the ART1 (Carpenter-Grossberg) network.

Pipeline:
    1. Load raw CSV (career_data.csv)
    2. Normalize academic scores (60-100) / 10 -> [6.0, 10.0]
    3. One-hot encode each feature into 10 bins
    4. Filter null vectors (all zeros)
    5. Shuffle and split into dataset_A / dataset_B
    6. Save binarized CSVs

Trabajo Practico Final Integrador — Inteligencia Artificial 2026
UADER — IDTI Lab de Redes Neuronales — Grupo 6

Usage:
    python preprocessor.py
    python preprocessor.py --input career_data.csv --output-dir data
"""

import csv
import os
import sys
import argparse
import numpy as np

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

FEATURES = [
    # Academic scores: original range 60-100, normalize by /10 -> [6, 10]
    {"name": "Math_Score",          "normalize": True},
    {"name": "Science_Score",       "normalize": True},
    # Skills: already in small integer range (1-5)
    {"name": "Programming_Skill",   "normalize": False},
    {"name": "Communication_Skill", "normalize": False},
    {"name": "Logical_Ability",     "normalize": False},
    # RIASEC scores: range 0-10
    {"name": "R_score",             "normalize": False},
    {"name": "I_score",             "normalize": False},
    {"name": "A_score",             "normalize": False},
    {"name": "S_score",             "normalize": False},
    {"name": "E_score",             "normalize": False},
    {"name": "C_score",             "normalize": False},
]

N_BINS = 10  # Discrete bins per feature: [0,1), [1,2), ..., [9,10]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw_csv(filepath):
    """
    Load the raw RIASEC career prediction CSV.

    Expected columns:
        Math_Score, Science_Score, Programming_Skill, Communication_Skill,
        Logical_Ability, R_score, I_score, A_score, S_score, E_score,
        C_score, Career

    Returns
    -------
    data : list of list of float
        11 numeric feature values per record.
    labels : list of str
        Career label for each record.
    headers : list of str
        Column headers from the CSV.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    data = []
    labels = []

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)

        for row_num, row in enumerate(reader, start=2):
            if not row or all(cell.strip() == "" for cell in row):
                continue
            try:
                values = [float(v) for v in row[:-1]]
                label = row[-1].strip()
                data.append(values)
                labels.append(label)
            except (ValueError, IndexError) as exc:
                print(f"Warning: skipping row {row_num}: {exc}")

    return data, labels, headers


# ---------------------------------------------------------------------------
# Binarization
# ---------------------------------------------------------------------------

def normalize_value(value, feature_def):
    """
    Normalize a single feature value.

    Academic scores (60-100) are divided by 10 to map into [6, 10].
    Other features are left unchanged.
    """
    if feature_def["normalize"]:
        return value / 10.0
    return float(value)


def value_to_onehot(value, n_bins=N_BINS):
    """
    Map a numeric value to a one-hot binary vector of length n_bins.

    Bin assignment:
        bin_idx = floor(value), clamped to [0, n_bins - 1]

    Examples:
        value=8.8, n_bins=10  ->  bin 8  ->  [0,0,0,0,0,0,0,0,1,0]
        value=0.0, n_bins=10  ->  bin 0  ->  [1,0,0,0,0,0,0,0,0,0]
        value=10.0, n_bins=10 ->  bin 9  ->  [0,0,0,0,0,0,0,0,0,1]
    """
    vec = [0] * n_bins
    idx = max(0, min(int(value), n_bins - 1))
    vec[idx] = 1
    return vec


def binarize_record(values, feature_defs=FEATURES, n_bins=N_BINS):
    """
    Convert one raw record to a binary vector.

    Each of the 11 features is normalized (if needed) and then
    one-hot encoded into n_bins positions.

    Returns
    -------
    binary_vector : list of int
        Length = len(feature_defs) * n_bins  (default: 11 * 10 = 110).
    """
    binary = []
    for i, feat in enumerate(feature_defs):
        norm = normalize_value(values[i], feat)
        binary.extend(value_to_onehot(norm, n_bins))
    return binary


def generate_feature_names(feature_defs=FEATURES, n_bins=N_BINS):
    """
    Generate column names for the binarized output.

    Format: 'FeatureName_bin0', 'FeatureName_bin1', ...
    """
    names = []
    for feat in feature_defs:
        for b in range(n_bins):
            names.append(f"{feat['name']}_bin{b}")
    return names


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def preprocess(input_path, output_dir="data", split_ratio=0.5,
               seed=42, verbose=True):
    """
    Run the complete preprocessing pipeline.

    Steps:
        1. Load raw data
        2. Normalize + binarize each record
        3. Filter null vectors
        4. Shuffle with fixed seed, split into A / B
        5. Save to output_dir/dataset_A.csv and dataset_B.csv

    Parameters
    ----------
    input_path : str
        Path to career_data.csv.
    output_dir : str
        Directory for output files.
    split_ratio : float
        Fraction of data going to dataset_A.
    seed : int
        Random seed for reproducibility.
    verbose : bool
        Print progress.

    Returns
    -------
    data_a, data_b : ndarray
        Binarized datasets.
    labels_a, labels_b : list of str
        Career labels.
    """
    if verbose:
        print(f"Loading raw data: {input_path}")
    data, labels, headers = load_raw_csv(input_path)
    if verbose:
        print(f"  Records  : {len(data)}")
        print(f"  Features : {len(data[0])}")
        print(f"  Careers  : {sorted(set(labels))}")

    # Binarize
    if verbose:
        print("Binarizing (normalize -> one-hot bins)...")

    binary_data = []
    binary_labels = []
    null_count = 0

    for values, label in zip(data, labels):
        bvec = binarize_record(values)
        if sum(bvec) == 0:
            null_count += 1
            continue
        binary_data.append(bvec)
        binary_labels.append(label)

    if verbose:
        print(f"  Binarizados: {len(binary_data)} vectores")
        print(f"  Dimensión: {len(binary_data[0])}")
        print(f"  Vectores nulos filtrados: {null_count}")

    # Shuffle + split
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(binary_data))
    split_pt = int(len(indices) * split_ratio)

    idx_a = sorted(indices[:split_pt])
    idx_b = sorted(indices[split_pt:])

    dataset_a = [binary_data[i] for i in idx_a]
    labels_a = [binary_labels[i] for i in idx_a]
    dataset_b = [binary_data[i] for i in idx_b]
    labels_b = [binary_labels[i] for i in idx_b]

    if verbose:
        print(f"  Dataset A: {len(dataset_a)} registros")
        print(f"  Dataset B: {len(dataset_b)} registros")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    feat_names = generate_feature_names()

    path_a = os.path.join(output_dir, "dataset_A.csv")
    path_b = os.path.join(output_dir, "dataset_B.csv")

    _save_csv(path_a, dataset_a, labels_a, feat_names)
    _save_csv(path_b, dataset_b, labels_b, feat_names)

    if verbose:
        print(f"\nGuardado:")
        print(f"  {path_a}")
        print(f"  {path_b}")

    return (
        np.array(dataset_a, dtype=int),
        np.array(dataset_b, dtype=int),
        labels_a,
        labels_b,
    )


def _save_csv(filepath, data, labels, feature_names):
    """Write binarized data + labels to CSV."""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(feature_names + ["Career"])
        for row, label in zip(data, labels):
            writer.writerow(list(row) + [label])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocesamiento de datos de carreras RIASEC para red ART1."
    )
    parser.add_argument(
        "--input", "-i", default="career_data.csv",
        help="Ruta al CSV original. Por defecto: career_data.csv"
    )
    parser.add_argument(
        "--output-dir", "-o", default="data",
        help="Directorio de salida. Por defecto: data/"
    )
    parser.add_argument(
        "--split", "-s", type=float, default=0.5,
        help="Proporción de división para Dataset A. Por defecto: 0.5"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Semilla aleatoria. Por defecto: 42"
    )

    args = parser.parse_args()
    preprocess(args.input, args.output_dir, args.split, args.seed)


if __name__ == "__main__":
    main()
