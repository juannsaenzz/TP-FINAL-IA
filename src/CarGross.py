import numpy as np
import pandas as pd
import argparse
import sys
import traceback

MANUAL_PAGE = """
=============================================================================
                          MANUAL: CarGross.py
                        Modelo ART1 en Python
=============================================================================

SINOPSIS
    python CarGross.py --input <archivo.csv> [--rho <valor>] [--L <valor>]

DESCRIPCIÓN
    CarGross.py es una implementación de la red neuronal de Teoría de Resonancia 
    Adaptativa 1 (ART1) propuesta por Carpenter y Grossberg. Este modelo de 
    aprendizaje no supervisado procesa vectores binarios y genera agrupamientos 
    (clústeres) de manera incremental, resolviendo el dilema de estabilidad y 
    plasticidad.

OPCIONES
    --input, -i     (Requerido) Ruta al archivo CSV con los datos binarios.
                    Se espera que la última columna sea el target ('Career'),
                    la cual será excluida del entrenamiento.
    --rho, -p       (Opcional) Parámetro de vigilancia (0.0 a 1.0). Default: 0.5
                    Controla la granularidad de los clústeres. Valores altos
                    generan muchas categorías específicas; valores bajos 
                    generan pocas categorías generales.
    --L, -l         (Opcional) Parámetro L (> 1.0). Default: 2.0
                    Sesga la búsqueda hacia categorías establecidas.
    --help, -h      Muestra este manual de ayuda y finaliza la ejecución.

EJEMPLOS DE USO
    Ejecución normal con valores por defecto:
        python CarGross.py -i ../data/dataset_A.csv
    
    Ejecución con alta vigilancia (más categorías):
        python CarGross.py -i ../data/dataset_B.csv --rho 0.8

ERRORES COMUNES
    - "No se encontró el archivo": Verifique que la ruta al CSV sea correcta.
    - "Vectores no binarios": La red ART1 requiere estrictamente 0s y 1s.
    - "Vectores nulos": Un vector de puros ceros anula la comparación.
=============================================================================
"""

class ART1:
    """
    Implementación de la red neuronal ART1 (Adaptive Resonance Theory).
    Autodocumentado y con lógica matricial.
    """
    def __init__(self, num_features, rho=0.5, L=2.0):
        self.num_features = num_features
        self.rho = rho
        self.L = L
        
        # Almacenaremos los pesos como listas dinámicas de vectores de numpy
        self.W_BU = [] # Bottom-Up: Tamaño (num_categorias, num_features)
        self.W_TD = [] # Top-Down: Tamaño (num_categorias, num_features)
        
    def _crear_categoria(self, x):
        """Crea una nueva categoría inicializando los pesos con el vector de entrada."""
        # Top-Down = x
        self.W_TD.append(x.copy())
        
        # Bottom-Up = (L * x) / (L - 1 + norma(x))
        norma_x = np.sum(x)
        bu_weights = (self.L * x) / (self.L - 1 + norma_x)
        self.W_BU.append(bu_weights)
        
        return len(self.W_TD) - 1

    def predecir(self, x):
        """
        Procesa un vector x a través de la red ART1.
        Retorna el índice del clúster al que fue asignado.
        """
        norma_x = np.sum(x)
        if norma_x == 0:
            raise ValueError("El vector de entrada es completamente nulo (0s). ART1 requiere vectores con norma > 0.")
            
        # Si la red está vacía, crear la primera categoría
        if len(self.W_TD) == 0:
            return self._crear_categoria(x)
            
        # Calcular activación de las categorías existentes (Capa F2)
        # T = W_BU dot X
        T = np.dot(self.W_BU, x)
        
        # Ordenar categorías por nivel de activación (descendente)
        categorias_ordenadas = np.argsort(T)[::-1]
        
        # Proceso de búsqueda por resonancia
        for j in categorias_ordenadas:
            # Expectativa Top-Down: intersección entre el vector x y el prototipo
            V = np.bitwise_and(x, self.W_TD[j])
            norma_v = np.sum(V)
            
            # Evaluación del Mecanismo de Reset (Vigilancia)
            similitud = norma_v / norma_x
            if similitud >= self.rho:
                # ¡Resonancia! Actualizamos pesos y asignamos a esta categoría
                self.W_TD[j] = V
                self.W_BU[j] = (self.L * V) / (self.L - 1 + norma_v)
                return j
                
        # Si ninguna categoría supera la vigilancia (Reset), creamos una nueva
        return self._crear_categoria(x)

    def entrenar(self, X):
        """
        Entrena la red con un conjunto de datos (lista de vectores o matriz 2D).
        """
        clusters = []
        for i, x in enumerate(X):
            cluster_id = self.predecir(x)
            clusters.append(cluster_id)
        return clusters

def run():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--input', '-i', type=str)
    parser.add_argument('--rho', '-p', type=float, default=0.5)
    parser.add_argument('--L', '-l', type=float, default=2.0)
    parser.add_argument('--help', '-h', action='store_true')
    
    try:
        args = parser.parse_args()
    except:
        print(MANUAL_PAGE)
        sys.exit(1)
        
    if args.help or not args.input:
        print(MANUAL_PAGE)
        sys.exit(0 if args.help else 1)
        
    try:
        print(f"[*] Cargando archivo: {args.input}")
        df = pd.read_csv(args.input)
        
        if 'Career' in df.columns:
            X = df.drop(columns=['Career']).values
            labels = df['Career'].values
        else:
            X = df.values
            labels = None
            
        if not np.array_equal(X, X.astype(bool)):
            raise ValueError("Los datos de entrada no son estrictamente binarios (ceros y unos).")
            
        print(f"[*] Datos cargados: {X.shape[0]} vectores de {X.shape[1]} bits.")
        print(f"[*] Inicializando ART1 (Vigilancia rho={args.rho}, L={args.L})")
        
        red = ART1(num_features=X.shape[1], rho=args.rho, L=args.L)
        
        print("[*] Procesando datos...")
        clusters = red.entrenar(X)
        
        n_clusters = len(red.W_TD)
        print(f"[+] Ejecución finalizada. Se formaron {n_clusters} categorías (clústeres).")
        
        # Mostrar resumen simple de distribución
        cluster_counts = pd.Series(clusters).value_counts().sort_index()
        for c_id, count in cluster_counts.items():
            print(f"    - Categoría {c_id}: {count} alumnos")
            
    except Exception as e:
        print("\n[!] OCURRIÓ UN ERROR DURANTE LA EJECUCIÓN:")
        print(f"    {str(e)}")
        print("\nPara más información sobre el uso correcto, lea el siguiente manual:")
        print(MANUAL_PAGE)
        sys.exit(1)

if __name__ == '__main__':
    run()
