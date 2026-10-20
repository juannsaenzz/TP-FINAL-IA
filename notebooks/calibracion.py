"""
run_demo.py — Complete ART1 Demo on RIASEC Career Data

Executes the full pipeline:
    1. Preprocess career_data.csv -> binary datasets A and B
    2. Sweep rho values on dataset A to find optimal vigilance
    3. Train ART1 on dataset A with optimal rho
    4. Train ART1 on dataset B with the same rho
    5. Analyze and compare clusters against original career labels
    6. Display prototype interpretation

Trabajo Practico Final Integrador — Inteligencia Artificial 2026
UADER — IDTI Lab de Redes Neuronales — Grupo 6

Usage:
    python run_demo.py
"""

import os
import sys
import argparse
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, os.path.join(project_root, "src"))

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from preprocessor import (
    preprocess,
    generate_feature_names,
    FEATURES,
    N_BINS,
)
from CarGross import ART1, load_binary_csv



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_header(title, char="="):
    """Print a formatted section header."""
    width = 70
    print()
    print(char * width)
    print(f"  {title}")
    print(char * width)


def career_distribution(labels):
    """Count occurrences of each career label."""
    dist = {}
    for lbl in labels:
        dist[lbl] = dist.get(lbl, 0) + 1
    return dict(sorted(dist.items(), key=lambda kv: -kv[1]))


def cluster_label_crosstab(assignments, labels):
    """
    Build a cross-tabulation of ART1 clusters vs original career labels.

    Returns dict: {cluster_id: {career_label: count}}
    """
    tab = {}
    for cat, lbl in zip(assignments, labels):
        cat = int(cat)
        tab.setdefault(cat, {})
        tab[cat][lbl] = tab[cat].get(lbl, 0) + 1
    return tab


def print_crosstab(tab, full=False, max_show=5):
    """Print a cluster-vs-label cross-tabulation."""
    if full:
        cids = sorted(tab.keys())
        print(f"\n  [Desglose completo: {len(cids)} categorías]")
    else:
        # Sort by population descending
        sorted_by_size = sorted(tab.keys(), key=lambda c: sum(tab[c].values()), reverse=True)
        cids = sorted_by_size[:max_show]
        print(f"\n  [Resumen Ejecutivo: Mostrando los {len(cids)} clústeres más poblados de {len(tab)}. Use --full para ver todos]")

    for cid in cids:
        if cid == -1:
            header = "SIN ASIGNAR"
        else:
            header = f"Cluster {cid}"
        total = sum(tab[cid].values())
        print(f"\n  {header} ({total} patrones):")

        for lbl, cnt in sorted(tab[cid].items(), key=lambda kv: -kv[1]):
            pct = 100.0 * cnt / total
            bar = "#" * int(pct / 2)
            print(f"    {lbl:25s} {cnt:4d} ({pct:5.1f}%) {bar}")


def interpret_prototype(prototype, feature_defs=FEATURES, n_bins=N_BINS):
    """
    Interpret a binary prototype vector in terms of active feature bins.

    Returns a dict: {feature_name: list of active bin indices}
    """
    interp = {}
    for i, feat in enumerate(feature_defs):
        start = i * n_bins
        end = start + n_bins
        segment = prototype[start:end]
        active_bins = [b for b in range(n_bins) if segment[b] > 0.5]
        if active_bins:
            interp[feat["name"]] = active_bins
    return interp


def print_prototype_interpretation(net, feature_defs=FEATURES,
                                   n_bins=N_BINS, full=False, max_show=5):
    """Print human-readable interpretation of learned prototypes."""
    protos = net.get_prototypes()
    n_show = net.n_committed if full else min(net.n_committed, max_show)

    if full:
        print(f"\n  Mostrando los {n_show} prototipos aprendidos:\n")
    else:
        print(f"\n  [Resumen: Mostrando los primeros {n_show} prototipos de {net.n_committed}. Use --full para ver todos]\n")

    for j in range(n_show):
        interp = interpret_prototype(protos[j], feature_defs, n_bins)
        active_count = int(np.sum(protos[j]))
        print(f"  Prototipo {j} ({active_count} bits activos):")

        for fname, bins in interp.items():
            bin_str = ", ".join(str(b) for b in bins)
            print(f"    {fname:25s} -> bins [{bin_str}]")

        print()



# ---------------------------------------------------------------------------
# Rho sweep
# ---------------------------------------------------------------------------

def rho_sweep(data, rho_values, L=2.0, max_categories=200, verbose=True):
    """
    Train ART1 with different rho values and collect metrics.

    Returns list of dicts with keys:
        rho, n_categories, n_unassigned, epochs
    """
    results = []

    for rho in rho_values:
        net = ART1(
            n_features=data.shape[1],
            rho=rho,
            L=L,
            max_categories=max_categories,
        )
        assignments, n_epochs = net.train(data, verbose=False)

        n_cats = net.n_committed
        n_unasgn = int(np.sum(assignments == -1))

        results.append({
            "rho": rho,
            "n_categories": n_cats,
            "n_unassigned": n_unasgn,
            "epochs": n_epochs,
        })

        if verbose:
            print(
                f"  rho={rho:.2f} | "
                f"{n_cats:3d} categorías | "
                f"{n_unasgn:3d} sin asignar | "
                f"{n_epochs:2d} épocas"
            )

    return results


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Simulación y calibración experimental de red ART1.")
    parser.add_argument("--full", "-f", action="store_true", help="Mostrar desglose exhaustivo de todos los clústeres.")
    args = parser.parse_args()

    # ==================================================================
    # Step 1: Preprocess
    # ==================================================================

    print_header("PASO 1: Preprocesamiento de career_data.csv")

    csv_path = os.path.join(project_root, "career_data.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(project_root, "archive", "career_data.csv")

    if not os.path.exists(csv_path):
        print(f"Error: no se encontró {csv_path}.", file=sys.stderr)
        print("Ubique career_data.csv en la raíz del proyecto.")
        sys.exit(1)

    data_dir = os.path.join(project_root, "data")


    data_a, data_b, labels_a, labels_b = preprocess(
        csv_path, output_dir=data_dir, split_ratio=0.5, seed=42
    )

    print("\n  Distribución de carreras en Dataset A:")
    for lbl, cnt in career_distribution(labels_a).items():
        print(f"    {lbl:25s} {cnt:4d}")

    print("\n  Distribución de carreras en Dataset B:")
    for lbl, cnt in career_distribution(labels_b).items():
        print(f"    {lbl:25s} {cnt:4d}")

    # ==================================================================
    # Step 2: Rho sweep on dataset A
    # ==================================================================
    print_header("PASO 2: Barrido del Parámetro de Vigilancia (rho) — Dataset A")
    print("  Evaluando cómo influye rho en la granularidad de categorías...\n")

    rho_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    sweep_results = rho_sweep(data_a, rho_values)

    # Pick a balanced rho: not too many, not too few categories.
    # Heuristic: closest to 7 categories (the dataset has ~7 careers).
    n_original_careers = len(set(labels_a))
    best = min(
        sweep_results,
        key=lambda r: abs(r["n_categories"] - n_original_careers)
    )
    optimal_rho = best["rho"]

    print(f"\n  Carreras originales en los datos: {n_original_careers}")
    print(f"  rho seleccionado: {optimal_rho:.2f} "
          f"({best['n_categories']} categorías)")

    # ==================================================================
    # Step 3: Full training on dataset A with optimal rho
    # ==================================================================
    print_header(
        f"PASO 3: Entrenamiento en Dataset A (rho={optimal_rho})"
    )

    net_a = ART1(
        n_features=data_a.shape[1],
        rho=optimal_rho,
        L=2.0,
        max_categories=200,
    )
    assignments_a, _ = net_a.train(data_a, verbose=True)

    print()
    net_a.summary()

    # Cross-tabulation
    print_header("Dataset A — Tabla Cruzada: Cluster vs Carrera", "-")
    tab_a = cluster_label_crosstab(assignments_a, labels_a)
    print_crosstab(tab_a, full=args.full)

    # Prototype interpretation
    print_header("Dataset A — Interpretación de Prototipos", "-")
    print_prototype_interpretation(net_a, full=args.full)

    # ==================================================================
    # Step 4: Full training on dataset B with same rho
    # ==================================================================
    print_header(
        f"PASO 4: Entrenamiento en Dataset B (rho={optimal_rho})"
    )

    net_b = ART1(
        n_features=data_b.shape[1],
        rho=optimal_rho,
        L=2.0,
        max_categories=200,
    )
    assignments_b, _ = net_b.train(data_b, verbose=True)

    print()
    net_b.summary()

    # Cross-tabulation
    print_header("Dataset B — Tabla Cruzada: Cluster vs Carrera", "-")
    tab_b = cluster_label_crosstab(assignments_b, labels_b)
    print_crosstab(tab_b, full=args.full)

    # Prototype interpretation
    print_header("Dataset B — Interpretación de Prototipos", "-")
    print_prototype_interpretation(net_b, full=args.full)

    # ==================================================================
    # Step 5: Comparison
    # ==================================================================
    print_header("PASO 5: Análisis Comparativo Consolidado")

    purities_a = [100.0 * max(tab_a[cid].values()) / sum(tab_a[cid].values()) for cid in tab_a if cid != -1]
    purities_b = [100.0 * max(tab_b[cid].values()) / sum(tab_b[cid].values()) for cid in tab_b if cid != -1]
    avg_p_a = float(np.mean(purities_a)) if purities_a else 0.0
    avg_p_b = float(np.mean(purities_b)) if purities_b else 0.0
    p100_a = [cid for cid in tab_a if cid != -1 and max(tab_a[cid].values()) == sum(tab_a[cid].values())]
    p100_b = [cid for cid in tab_b if cid != -1 and max(tab_b[cid].values()) == sum(tab_b[cid].values())]

    print(f"\n  {'Métrica':<30s} {'Dataset A':>12s} {'Dataset B':>12s}")
    print(f"  {'-'*30} {'-'*12} {'-'*12}")
    print(f"  {'Patrones de entrada':<30s} {len(labels_a):>12d} {len(labels_b):>12d}")
    print(f"  {'Categorías formadas':<30s} {net_a.n_committed:>12d} {net_b.n_committed:>12d}")
    print(f"  {'Patrones sin asignar':<30s} "
          f"{int(np.sum(assignments_a == -1)):>12d} "
          f"{int(np.sum(assignments_b == -1)):>12d}")
    print(f"  {'Vigilancia (rho)':<30s} {optimal_rho:>12.2f} {optimal_rho:>12.2f}")
    print(f"  {'Pureza promedio de clúster':<30s} {avg_p_a:>11.1f}% {avg_p_b:>11.1f}%")
    print(f"  {'Clústeres 100% monocolor':<30s} {len(p100_a):>12d} {len(p100_b):>12d}")

    if not args.full:
        print("\n  Ejemplos de Clústeres con 100% de Pureza Vocacional:")
        for cid in p100_a:
            dom = max(tab_a[cid], key=tab_a[cid].get)
            print(f"    - Dataset A (Cluster {cid:2d}): {dom:18s} ({sum(tab_a[cid].values())} alumnos)")
        for cid in p100_b:
            dom = max(tab_b[cid], key=tab_b[cid].get)
            print(f"    - Dataset B (Cluster {cid:2d}): {dom:18s} ({sum(tab_b[cid].values())} alumnos)")
        print("\n  [i] Para auditar los 79 clústeres completos, ejecute: python notebooks/calibracion.py --full")
    else:
        print(f"\n  Carrera dominante por cluster (Dataset A):")
        for cid in sorted(tab_a):
            if cid == -1:
                continue
            dominant = max(tab_a[cid], key=tab_a[cid].get)
            total = sum(tab_a[cid].values())
            purity = 100.0 * tab_a[cid][dominant] / total
            print(f"    Cluster {cid:3d}: {dominant:25s} ({purity:.1f}% pureza)")

        print(f"\n  Carrera dominante por cluster (Dataset B):")
        for cid in sorted(tab_b):
            if cid == -1:
                continue
            dominant = max(tab_b[cid], key=tab_b[cid].get)
            total = sum(tab_b[cid].values())
            purity = 100.0 * tab_b[cid][dominant] / total
            print(f"    Cluster {cid:3d}: {dominant:25s} ({purity:.1f}% pureza)")


    # Save prototypes
    proto_a_path = os.path.join(data_dir, "prototypes_A.csv")
    proto_b_path = os.path.join(data_dir, "prototypes_B.csv")
    feat_names = generate_feature_names()
    net_a.save_prototypes(proto_a_path, feat_names)
    net_b.save_prototypes(proto_b_path, feat_names)

    # ==================================================================
    # Summary
    # ==================================================================
    print_header("DEMO FINALIZADA")
    print(f"""
  Archivos generados:
    data/dataset_A.csv      — Dataset A binarizado ({len(labels_a)} registros)
    data/dataset_B.csv      — Dataset B binarizado ({len(labels_b)} registros)
    data/prototypes_A.csv   — Prototipos aprendidos del Dataset A
    data/prototypes_B.csv   — Prototipos aprendidos del Dataset B

  Para volver a ejecutar con diferentes parámetros:
    python src/CarGross.py -i data/dataset_A.csv -r 0.1
    python src/CarGross.py -i data/dataset_B.csv -r 0.1 -o data/resultados.csv
""")


if __name__ == "__main__":
    main()
