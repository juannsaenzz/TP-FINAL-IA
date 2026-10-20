import pandas as pd
import numpy as np
import sys
import os

# Manejo dinámico de rutas (soporta CLI desde root, desde notebooks, o interactivo en Jupyter)
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.path.abspath(os.getcwd())
    if not script_dir.endswith('notebooks'):
        script_dir = os.path.join(script_dir, 'notebooks')

project_root = os.path.abspath(os.path.join(script_dir, '..'))

# Agregamos la ruta src al path para importar CarGross
sys.path.append(os.path.join(project_root, 'src'))
from CarGross import ART1

def run_calibration():
    print("=== Experimento de Calibración de Parámetro de Vigilancia (p) ===")
    
    data_path = os.path.join(project_root, 'data', 'dataset_A.csv')
    df = pd.read_csv(data_path)
    
    X = df.drop(columns=['Career']).values
    labels = df['Career'].values
    
    vigilance_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for rho in vigilance_values:
        print(f"\n[*] Ejecutando ART1 con rho = {rho}")
        red = ART1(num_features=X.shape[1], rho=rho)
        clusters = red.entrenar(X)
        
        n_clusters = len(red.W_TD)
        print(f"    -> Clústeres formados: {n_clusters}")
        
        # Analizamos la coherencia psicopedagógica del Clúster más grande
        df_results = pd.DataFrame({'Cluster': clusters, 'Career': labels})
        biggest_cluster = df_results['Cluster'].value_counts().idxmax()
        
        careers_in_cluster = df_results[df_results['Cluster'] == biggest_cluster]['Career'].value_counts()
        print(f"    -> Composición del clúster más grande (ID {biggest_cluster}):")
        for career, count in careers_in_cluster.items():
            print(f"       - {career}: {count}")

if __name__ == "__main__":
    run_calibration()
