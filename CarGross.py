"""
CarGross.py — ART1 Neural Network (Adaptive Resonance Theory 1)
Carpenter-Grossberg Network Implementation

Unsupervised clustering of binary patterns using a two-layer
competitive architecture with vigilance-modulated reset mechanism.

Trabajo Practico Final Integrador — Inteligencia Artificial 2026
UADER — IDTI Lab de Redes Neuronales
Docentes: Dra. Daniela Lopez De Luise & Alejandro Perez Museo
Grupo 6

Usage:
    python CarGross.py --input data/dataset_A.csv --rho 0.5
    python CarGross.py --input data/dataset_B.csv --rho 0.7 --output results.csv
    python CarGross.py --help
"""

import numpy as np
import csv
import sys
import os
import argparse

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# ART1 Network
# ---------------------------------------------------------------------------

class ART1:
    """
    Adaptive Resonance Theory 1 (ART1) Neural Network.

    Architecture designed by Carpenter & Grossberg (1987) for unsupervised
    clustering of binary input patterns. Consists of:

    - F1 (Comparison Layer): receives binary input, computes match with
      stored top-down prototypes.
    - F2 (Recognition Layer): competitive layer where each committed node
      stores one learned category via its top-down weight vector.
    - Reset Module: compares match ratio against vigilance parameter rho.
      On failure, inhibits the current winner and forces a new search.

    Parameters
    ----------
    n_features : int
        Dimension of binary input vectors.
    rho : float, default=0.5
        Vigilance parameter in [0, 1]. Controls cluster granularity.
        Low rho (~0.1-0.3) = few broad clusters (under-classification).
        High rho (~0.7-0.9) = many specific clusters (over-classification).
    L : float, default=2.0
        Choice parameter (must be > 1). Normalizes bottom-up weights.
    max_categories : int, default=200
        Maximum number of categories the F2 layer can allocate.
    max_epochs : int, default=100
        Maximum training passes over the full dataset.

    Attributes
    ----------
    weights_bu : ndarray of shape (n_features, max_categories)
        Bottom-up weights (F1 -> F2). Initialized to L / (L - 1 + n).
    weights_td : ndarray of shape (max_categories, n_features)
        Top-down weights (F2 -> F1). Each row is a category prototype.
        Initialized to all ones.
    n_committed : int
        Number of currently committed (learned) categories.
    """

    def __init__(self, n_features, rho=0.5, L=2.0,
                 max_categories=200, max_epochs=100):
        if n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {n_features}")
        if not 0.0 <= rho <= 1.0:
            raise ValueError(f"rho must be in [0, 1], got {rho}")
        if L <= 1.0:
            raise ValueError(f"L must be > 1, got {L}")
        if max_categories < 1:
            raise ValueError(
                f"max_categories must be >= 1, got {max_categories}"
            )

        self.n_features = n_features
        self.rho = rho
        self.L = L
        self.max_categories = max_categories
        self.max_epochs = max_epochs
        self.n_committed = 0
        self.categories = np.array([], dtype=int)
        self._history = []

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialization
    # ------------------------------------------------------------------

    def _init_weights(self):
        """
        Initialize weight matrices.

        Bottom-up (F1->F2): b_ij = L / (L - 1 + n)
            Ensures an uncommitted node wins only when no committed
            node matches well enough.

        Top-down (F2->F1): t_ji = 1 for all j, i
            A new category starts as "accept all" and specializes
            through learning (AND operation with presented patterns).
        """
        n = self.n_features
        M = self.max_categories

        self.weights_bu = np.full(
            (n, M), self.L / (self.L - 1.0 + n), dtype=np.float64
        )
        self.weights_td = np.ones((M, n), dtype=np.float64)

    # ------------------------------------------------------------------
    # Core ART1 operations
    # ------------------------------------------------------------------

    def _compute_activations(self, x, inhibited):
        """
        Compute F2 net input for all available nodes.

        y_j = sum_i( b_ij * x_i ) for each non-inhibited node j.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            Binary input vector.
        inhibited : set of int
            Indices of inhibited F2 nodes.

        Returns
        -------
        activations : ndarray
            Net inputs. Inhibited nodes get -inf.
        """
        n_active = min(self.n_committed + 1, self.max_categories)
        activations = x @ self.weights_bu[:, :n_active]

        for idx in inhibited:
            if idx < n_active:
                activations[idx] = -np.inf

        return activations

    def _vigilance_test(self, x, j):
        """
        Perform vigilance test for category j against input x.

        Computes the match ratio: |T_j AND x| / |x|.
        Resonance occurs when this ratio >= rho.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
        j : int — category index.

        Returns
        -------
        passes : bool
        match_ratio : float
        match_vector : ndarray — element-wise AND result.
        """
        match_vector = np.minimum(self.weights_td[j], x)
        norm_x = np.sum(x)

        if norm_x == 0:
            return False, 0.0, match_vector

        match_ratio = np.sum(match_vector) / norm_x
        return match_ratio >= self.rho, match_ratio, match_vector

    def _update_weights(self, x, j):
        """
        Update weights after resonance with category j.

        Top-down:  T_j_new = T_j_old AND x  (intersection)
        Bottom-up: b_ij_new = (T_j AND x)_i / (L + |T_j AND x|)

        This makes the prototype more specific: only features
        consistently present across all resonating patterns survive.
        """
        match = np.minimum(self.weights_td[j], x)
        self.weights_td[j] = match
        norm_match = np.sum(match)
        self.weights_bu[:, j] = match / (self.L + norm_match)

    # ------------------------------------------------------------------
    # Single-pattern training
    # ------------------------------------------------------------------

    def train_pattern(self, x):
        """
        Present one binary pattern and learn.

        Full ART1 cycle:
        1. Competition: compute F2 activations, pick winner.
        2. Comparison: read out top-down prototype, AND with input.
        3. Vigilance test: pass -> resonance + weight update.
                           fail -> reset (inhibit winner), repeat.
        4. If all nodes exhausted, allocate a new category.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            Binary input vector (0/1 values).

        Returns
        -------
        category : int
            Assigned category index. -1 if unassignable (null vector
            or max categories reached).
        """
        x = np.asarray(x, dtype=np.float64)

        if x.shape[0] != self.n_features:
            raise ValueError(
                f"Input dim {x.shape[0]} != expected {self.n_features}"
            )

        if np.sum(x) == 0:
            return -1  # Null vector — cannot classify

        inhibited = set()

        while True:
            activations = self._compute_activations(x, inhibited)

            # All nodes inhibited?
            if np.all(np.isinf(activations) & (activations < 0)):
                break

            J = int(np.argmax(activations))

            passes, _, _ = self._vigilance_test(x, J)

            if passes:
                is_new = J >= self.n_committed
                self._update_weights(x, J)
                if is_new:
                    self.n_committed += 1
                return J

            # Reset: inhibit J, keep searching
            inhibited.add(J)

            n_active = min(self.n_committed + 1, self.max_categories)
            if len(inhibited) >= n_active:
                break

        # All exhausted — try to create a new category
        if self.n_committed < self.max_categories:
            J_new = self.n_committed
            self.weights_td[J_new] = x.copy()
            norm_x = np.sum(x)
            self.weights_bu[:, J_new] = x / (self.L + norm_x)
            self.n_committed += 1
            return J_new

        return -1  # Max categories reached

    # ------------------------------------------------------------------
    # Full-dataset training
    # ------------------------------------------------------------------

    def train(self, X, verbose=True):
        """
        Train the network on a complete dataset.

        Iterates over all patterns repeatedly until category
        assignments stabilize (no changes) or max_epochs is reached.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Binary input matrix.
        verbose : bool
            Print per-epoch progress.

        Returns
        -------
        assignments : ndarray of shape (n_samples,)
            Final category index for each pattern.
        n_epochs : int
            Number of epochs completed.
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]

        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Input dim {X.shape[1]} != expected {self.n_features}"
            )

        assignments = np.full(n_samples, -1, dtype=int)

        for epoch in range(1, self.max_epochs + 1):
            changes = 0

            for i in range(n_samples):
                cat = self.train_pattern(X[i])
                if cat != assignments[i]:
                    changes += 1
                assignments[i] = cat

            info = {
                "epoch": epoch,
                "changes": changes,
                "categories": self.n_committed,
                "unassigned": int(np.sum(assignments == -1)),
            }
            self._history.append(info)

            if verbose:
                print(
                    f"  Época {epoch:3d} | "
                    f"{changes:5d} cambios | "
                    f"{self.n_committed:3d} categorías | "
                    f"{info['unassigned']:3d} sin asignar"
                )

            if changes == 0:
                if verbose:
                    print(f"  Convergencia alcanzada tras {epoch} época(s).")
                break
        else:
            if verbose:
                print(
                    f"  Advertencia: no se alcanzó convergencia en "
                    f"{self.max_epochs} épocas."
                )

        self.categories = assignments
        return assignments, epoch

    # ------------------------------------------------------------------
    # Prediction (no weight update)
    # ------------------------------------------------------------------

    def predict(self, x):
        """
        Classify a single pattern against learned categories.

        Does NOT update weights — purely inference.

        Returns
        -------
        category : int — best matching category, or -1.
        match_ratio : float — similarity to the winning prototype.
        """
        x = np.asarray(x, dtype=np.float64)
        if np.sum(x) == 0:
            return -1, 0.0

        if self.n_committed == 0:
            return -1, 0.0

        activations = x @ self.weights_bu[:, :self.n_committed]
        order = np.argsort(-activations)

        for J in order:
            passes, ratio, _ = self._vigilance_test(x, int(J))
            if passes:
                return int(J), ratio

        return -1, 0.0

    def predict_batch(self, X):
        """Classify multiple patterns. Returns (categories, match_ratios)."""
        X = np.asarray(X, dtype=np.float64)
        cats = np.zeros(X.shape[0], dtype=int)
        ratios = np.zeros(X.shape[0], dtype=np.float64)
        for i in range(X.shape[0]):
            cats[i], ratios[i] = self.predict(X[i])
        return cats, ratios

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def get_prototypes(self):
        """Return top-down weight vectors for committed categories."""
        return self.weights_td[:self.n_committed].copy()

    def get_category_sizes(self, assignments=None):
        """Count patterns per category. Returns dict {cat_id: count}."""
        if assignments is None:
            assignments = self.categories
        sizes = {}
        for c in assignments:
            c = int(c)
            sizes[c] = sizes.get(c, 0) + 1
        return sizes

    def get_training_history(self):
        """Return list of per-epoch dicts (epoch, changes, categories, unassigned)."""
        return list(self._history)

    def summary(self):
        """Print a human-readable summary of the network state."""
        sep = "=" * 60
        print(sep)
        print("  Resumen de Red ART1 (Carpenter-Grossberg)")
        print(sep)
        print(f"  Dimensión de entrada : {self.n_features}")
        print(f"  Vigilancia (rho)     : {self.rho}")
        print(f"  Parámetro L          : {self.L}")
        print(f"  Categorías máximas   : {self.max_categories}")
        print(f"  Categorías creadas   : {self.n_committed}")

        if len(self.categories) > 0:
            sizes = self.get_category_sizes()
            unassigned = sizes.pop(-1, 0)
            print(f"  Patrones entrenados  : {len(self.categories)}")
            print(f"  Sin asignar          : {unassigned}")
            if sizes:
                print(f"  Tamaño de categorías :")
                for cid in sorted(sizes):
                    print(f"    Cat {cid:3d} : {sizes[cid]:5d} patrones")
        print(sep)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_prototypes(self, filepath, feature_names=None):
        """
        Save learned prototypes to CSV.

        Each row is one committed category with its binary prototype
        vector and category index.
        """
        if self.n_committed == 0:
            print("Advertencia: no hay categorías para guardar.")
            return

        protos = self.get_prototypes()

        if feature_names and len(feature_names) == self.n_features:
            header = ["Category"] + list(feature_names)
        else:
            header = ["Category"] + [f"f{i}" for i in range(self.n_features)]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for j in range(self.n_committed):
                writer.writerow([j] + protos[j].astype(int).tolist())

        print(f"Prototipos guardados en: {filepath}")


# ---------------------------------------------------------------------------
# CSV Loader
# ---------------------------------------------------------------------------

def load_binary_csv(filepath):
    """
    Load a CSV of binary data.

    If the last column is non-numeric, it is treated as a label column.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features)
    headers : list of str — feature column names.
    labels : list of str or None — label for each row.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Archivo no encontrado: {filepath}")

    rows = []
    labels = []
    has_labels = False

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)

        for line_num, row in enumerate(reader, start=2):
            if not row or all(c.strip() == "" for c in row):
                continue
            try:
                vals = [float(v) for v in row]
                rows.append(vals)
                labels.append(None)
            except ValueError:
                # Last column is a label
                try:
                    vals = [float(v) for v in row[:-1]]
                    rows.append(vals)
                    labels.append(row[-1].strip())
                    has_labels = True
                except ValueError:
                    print(f"Advertencia: omitiendo fila malformada {line_num}")

    data = np.array(rows, dtype=np.float64)
    if has_labels:
        return data, headers[:-1], labels
    return data, headers, None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """Command-line interface for CarGross.py."""
    parser = argparse.ArgumentParser(
        prog="CarGross",
        description=(
            "CarGross — Red Neuronal ART1 (Carpenter-Grossberg).\n"
            "Agrupamiento no supervisado de patrones binarios."
        ),
        epilog=(
            "Ejemplos:\n"
            "  python CarGross.py -i data/dataset_A.csv -r 0.5\n"
            "  python CarGross.py -i data/dataset_B.csv -r 0.7 -o results.csv\n"
            "\n"
            "Diagnóstico de problemas (corridas infructuosas):\n"
            "  - ¿Demasiadas categorías?   Reducir rho.\n"
            "  - ¿Muy pocas categorías?    Aumentar rho.\n"
            "  - ¿Patrones sin asignar?    Revisar vectores nulos o reducir rho.\n"
            "  - ¿Valores no binarios?     Ejecutar preprocessor.py primero.\n"
            "  - ¿Convergencia lenta?      Aumentar --max-epochs.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input", "-i", required=True,
        help="Ruta al CSV de entrada con datos binarios (0/1)."
    )
    parser.add_argument(
        "--rho", "-r", type=float, default=0.5,
        help="Parámetro de vigilancia [0, 1]. Por defecto: 0.5"
    )
    parser.add_argument(
        "--L", type=float, default=2.0,
        help="Parámetro de elección (> 1). Por defecto: 2.0"
    )
    parser.add_argument(
        "--max-categories", "-m", type=int, default=200,
        help="Máximo de categorías F2. Por defecto: 200"
    )
    parser.add_argument(
        "--max-epochs", "-e", type=int, default=100,
        help="Máximo de épocas de entrenamiento. Por defecto: 100"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="CSV de salida para prototipos (opcional)."
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suprimir salida detallada época por época."
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    # ---- Validate ----
    if not 0.0 <= args.rho <= 1.0:
        print(f"Error: rho debe estar en [0, 1], se obtuvo {args.rho}", file=sys.stderr)
        sys.exit(1)
    if args.L <= 1.0:
        print(f"Error: L debe ser > 1, se obtuvo {args.L}", file=sys.stderr)
        sys.exit(1)

    # ---- Load ----
    try:
        print(f"Cargando: {args.input}")
        data, headers, labels = load_binary_csv(args.input)
        print(f"  {data.shape[0]} patrones x {data.shape[1]} características")
    except Exception as exc:
        print(f"Error al cargar datos: {exc}", file=sys.stderr)
        print(
            "\nDiagnóstico:\n"
            "  - Verificar que el archivo exista y sea un CSV válido.\n"
            "  - Los valores deben ser binarios (0/1).\n"
            "  - Usar preprocessor.py para binarizar datos originales.\n"
        )
        sys.exit(1)

    # ---- Check binary ----
    uniq = np.unique(data)
    non_bin = [v for v in uniq if v not in (0.0, 1.0)]
    if non_bin:
        print(
            f"Error: se encontraron valores no binarios: {non_bin[:10]}.\n"
            "ART1 requiere estrictamente entradas binarias (0/1).\n"
            "Usar preprocessor.py para convertir datos originales.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ---- Train ----
    print(f"\nEntrenando ART1 (rho={args.rho}, L={args.L})...\n")
    net = ART1(
        n_features=data.shape[1],
        rho=args.rho,
        L=args.L,
        max_categories=args.max_categories,
        max_epochs=args.max_epochs,
    )

    assignments, _ = net.train(data, verbose=not args.quiet)

    # ---- Summary ----
    print()
    net.summary()

    # ---- Cross-tab with labels ----
    if labels and any(l is not None for l in labels):
        print("\nDistribución de Categorías <-> Etiquetas Originales de Carrera:")
        print("-" * 55)
        cat_map = {}
        for cat, lbl in zip(assignments, labels):
            cat_map.setdefault(int(cat), {})
            cat_map[int(cat)][lbl] = cat_map[int(cat)].get(lbl, 0) + 1

        for cid in sorted(cat_map):
            tag = "Sin asignar" if cid == -1 else f"Categoría {cid}"
            print(f"  {tag}:")
            for lbl, cnt in sorted(
                cat_map[cid].items(), key=lambda kv: -kv[1]
            ):
                print(f"    {lbl:25s} {cnt:5d}")

    # ---- Save ----
    if args.output:
        net.save_prototypes(args.output, headers)

    return net, assignments


if __name__ == "__main__":
    main()
