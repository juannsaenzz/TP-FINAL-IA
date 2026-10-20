import pandas as pd
import numpy as np
import os

def binarize_row(row):
    # Vector de 110 bits (11 variables * 10 bines)
    bin_vector = np.zeros(110, dtype=int)
    
    # 1. Math_Score
    idx_math = min(9, int(row['Math_Score'] / 10))
    bin_vector[0 * 10 + idx_math] = 1
    
    # 2. Science_Score
    idx_sci = min(9, int(row['Science_Score'] / 10))
    bin_vector[1 * 10 + idx_sci] = 1
    
    # 3. Programming_Skill
    idx_prog = int(row['Programming_Skill'] * 2) - 1
    idx_prog = max(0, min(9, idx_prog)) # Safety clamp
    bin_vector[2 * 10 + idx_prog] = 1
    
    # 4. Communication_Skill
    idx_comm = int(row['Communication_Skill'] * 2) - 1
    idx_comm = max(0, min(9, idx_comm))
    bin_vector[3 * 10 + idx_comm] = 1
    
    # 5. Logical_Ability
    idx_log = int(row['Logical_Ability'] * 2) - 1
    idx_log = max(0, min(9, idx_log))
    bin_vector[4 * 10 + idx_log] = 1
    
    # 6. R_score
    idx_r = min(9, int(row['R_score']))
    bin_vector[5 * 10 + idx_r] = 1
    
    # 7. I_score
    idx_i = min(9, int(row['I_score']))
    bin_vector[6 * 10 + idx_i] = 1
    
    # 8. A_score
    idx_a = min(9, int(row['A_score']))
    bin_vector[7 * 10 + idx_a] = 1
    
    # 9. S_score
    idx_s = min(9, int(row['S_score']))
    bin_vector[8 * 10 + idx_s] = 1
    
    # 10. E_score
    idx_e = min(9, int(row['E_score']))
    bin_vector[9 * 10 + idx_e] = 1
    
    # 11. C_score
    idx_c = min(9, int(row['C_score']))
    bin_vector[10 * 10 + idx_c] = 1
    
    return pd.Series(bin_vector)

def preprocess():
    # Rutas
    input_file = 'archive/career_data.csv'
    output_a = 'data/dataset_A.csv'
    output_b = 'data/dataset_B.csv'
    
    if not os.path.exists(input_file):
        print(f"Error: No se encontró el archivo {input_file}")
        return

    print("Cargando dataset...")
    df = pd.read_csv(input_file)
    
    # Nombres de columnas para el output
    cols = ['Math_Score', 'Science_Score', 'Programming_Skill', 'Communication_Skill', 
            'Logical_Ability', 'R_score', 'I_score', 'A_score', 'S_score', 'E_score', 'C_score']
    
    bin_cols = []
    for col in cols:
        for i in range(1, 11):
            bin_cols.append(f"{col}_v{i}")
            
    print("Aplicando binarización y codificación dispersa (110 bits)...")
    df_bin = df.apply(binarize_row, axis=1)
    df_bin.columns = bin_cols
    
    # Agregamos la columna target (Career) al final para análisis posterior
    df_bin['Career'] = df['Career']
    
    # Validación: verificar que no haya vectores nulos (norma != 11)
    normas = df_bin[bin_cols].sum(axis=1)
    if (normas != 11).any():
        print("ADVERTENCIA: Se encontraron vectores con una norma distinta de 11. Revisar lógica.")
    else:
        print("Validación exitosa: Todos los vectores tienen 11 bits activos (cero vectores nulos).")
    
    # Mezclamos los datos y partimos en dos datasets (mínimo 50 según consigna)
    print("Particionando datasets...")
    df_bin = df_bin.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Partimos en dos pedazos de 100 registros para estar sobrados de la consigna (mínimo 50)
    dataset_a = df_bin.iloc[:100]
    dataset_b = df_bin.iloc[100:200]
    
    dataset_a.to_csv(output_a, index=False)
    dataset_b.to_csv(output_b, index=False)
    
    print(f"Dataset A guardado: {output_a} ({len(dataset_a)} registros)")
    print(f"Dataset B guardado: {output_b} ({len(dataset_b)} registros)")

if __name__ == '__main__':
    preprocess()
