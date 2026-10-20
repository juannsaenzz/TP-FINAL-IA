# CarGross — Red Neuronal ART1 (Carpenter-Grossberg)

Implementación en Python de una red neuronal ART1 (Adaptive Resonance Theory 1), diseñada por Stephen Grossberg y Gail Carpenter (1987), para agrupamiento no supervisado de patrones binarios.

**Trabajo Práctico Final Integrador — Inteligencia Artificial 2026**
UADER — IDTI Lab de Redes Neuronales
Docentes: Dra. Daniela López De Luise & Alejandro Pérez Museo
Grupo 6

---

## Tabla de Contenidos

1. [Alcances y Limitaciones](#alcances-y-limitaciones)
2. [Instalación](#instalación)
3. [Estructura del Proyecto](#estructura-del-proyecto)
4. [Modo de Uso](#modo-de-uso)
5. [Test Demo](#test-demo)
6. [Datasets](#datasets)
7. [Salida Esperada](#salida-esperada)
8. [Parámetros de la Red](#parámetros-de-la-red)
9. [FAQ](#faq)
10. [Referencias](#referencias)

---

## Alcances y Limitaciones

### Alcances

- Implementación completa del algoritmo ART1 para datos binarios.
- Preprocesamiento automático de datos RIASEC (normalización, binarización one-hot).
- Soporte para múltiples datasets CSV como entrada.
- Barrido automático del parámetro de vigilancia (ρ) para calibración.
- Análisis cruzado entre clusters ART1 y etiquetas de carrera originales.
- Interpretación legible de los prototipos aprendidos.
- Manejo robusto de errores y mensajes de ayuda ante corridas infructuosas.

### Limitaciones

- **Solo datos binarios**: ART1 trabaja exclusivamente con vectores de 0 y 1. Para datos continuos se requiere ART2 o Fuzzy ART (no implementados).
- **Sensibilidad al orden de presentación**: El orden en que se presentan los patrones puede afectar los clusters resultantes.
- **Sin interfaz gráfica**: La interacción es por línea de comandos.
- **Complejidad computacional**: Para datasets muy grandes (>10000 registros) el entrenamiento puede ser lento debido a la naturaleza secuencial del algoritmo.
- **Prototipos restrictivos**: Los prototipos se especializan por intersección (AND), lo que puede generar prototipos vacíos si los patrones dentro de un cluster son muy diversos.

---

## Instalación

### Requisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Pasos

1. **Clonar o descargar** el proyecto en una carpeta local.

2. **Instalar dependencias**:

```bash
pip install -r requirements.txt
```

Esto instala únicamente `numpy`.

3. **Verificar la instalación**:

```bash
python CarGross.py --help
```

Debería mostrar la ayuda del programa con todas las opciones disponibles.

---

## Estructura del Proyecto

```text
TPFI/
├── src/
│   ├── CarGross.py          # Implementación principal de la red ART1
│   └── preprocessor.py      # Preprocesamiento de datos RIASEC (110 bits)
├── notebooks/
│   └── calibracion.py       # Simulación experimental, barrido de rho y calibración completa
├── data/
│   ├── dataset_A.csv        # Dataset A binarizado (1.200 registros)
│   ├── dataset_B.csv        # Dataset B binarizado (1.200 registros)
│   ├── prototypes_A.csv     # Prototipos aprendidos del dataset A
│   └── prototypes_B.csv     # Prototipos aprendidos del dataset B
├── career_data.csv          # Dataset psicométrico original
├── INFORME.md               # Informe final con carátula institucional y fórmulas
├── MANUAL_REFERENCIA.md     # Manual de referencia técnica, alcances y limitaciones
├── requirements.txt         # Dependencias (numpy)
└── README.md                # Guía rápida del repositorio
```

---

## Modo de Uso

### 1. Preprocesar datos (generar datasets binarizados)

```bash
python src/preprocessor.py
```

Esto toma `career_data.csv` y genera `data/dataset_A.csv` y `data/dataset_B.csv`.

### 2. Entrenar la red ART1

```bash
python src/CarGross.py --input data/dataset_A.csv --rho 0.1
```

Opciones disponibles:

| Opción | Descripción | Default |
|--------|------------|---------|
| `--input`, `-i` | Archivo CSV de entrada (binario) | *requerido* |
| `--rho`, `-r` | Parámetro de vigilancia [0, 1] | 0.5 |
| `--L` | Parámetro de elección (> 1) | 2.0 |
| `--max-categories`, `-m` | Máximo de categorías F2 | 200 |
| `--max-epochs`, `-e` | Máximo de épocas | 100 |
| `--output`, `-o` | Archivo CSV para guardar prototipos | — |
| `--quiet`, `-q` | Suprimir progreso por época | — |

### 3. Ejecutar la calibración y demo completo

```bash
python notebooks/calibracion.py
```


Ejecuta automáticamente: preprocesamiento → barrido de ρ → entrenamiento en ambos datasets → análisis comparativo.

---

## Test Demo

Ejecutar:

```bash
python notebooks/calibracion.py
# O para reporte completo por categoría:
python notebooks/calibracion.py --full
```

El demo realizará:

1. **Preprocesamiento**: Lee `career_data.csv`, normaliza scores académicos (÷10), binariza cada feature en 10 bins one-hot (110 dimensiones binarias totales), filtra vectores nulos, y divide en dos datasets.

2. **Barrido de ρ**: Entrena con ρ = 0.1, 0.2, ..., 0.9 para encontrar el valor que genera un número de categorías cercano a las 7 carreras originales.

3. **Entrenamiento completo** en dataset A y B con el ρ óptimo.

4. **Análisis cruzado**: Tabla de clusters vs. carreras originales, mostrando la composición de cada cluster.

5. **Interpretación de prototipos**: Qué bins de qué features están activos en cada prototipo aprendido.

---

## Datasets

### Dataset original: `career_data.csv`

Fuente: [Kaggle — Student Career Prediction using RIASEC](https://www.kaggle.com/datasets/svenkateshkumar/student-career-prediction-using-riasec-dataset)

- **2401 registros** × 12 columnas
- **Variables**:
  - `Math_Score` (60-100): Rendimiento en matemáticas
  - `Science_Score` (60-100): Rendimiento en ciencias
  - `Programming_Skill` (1-5): Habilidad de programación
  - `Communication_Skill` (1-5): Habilidad de comunicación
  - `Logical_Ability` (1-5): Capacidad lógica
  - `R_score` (0-10): Perfil Realista (RIASEC)
  - `I_score` (0-10): Perfil Investigador
  - `A_score` (0-10): Perfil Artístico
  - `S_score` (0-10): Perfil Social
  - `E_score` (0-10): Perfil Emprendedor (comercial)
  - `C_score` (0-10): Perfil Convencional (organización)
  - `Career`: Carrera predicha (Entrepreneur, Accountant, Teacher, Software Engineer, Doctor, Data Scientist)

### Datasets binarizados

Tras el preprocesamiento, cada registro se convierte en un vector binario de **110 dimensiones** (11 features × 10 bins cada una).

- `data/dataset_A.csv`: ~1200 registros binarizados + etiqueta Career
- `data/dataset_B.csv`: ~1200 registros binarizados + etiqueta Career

---

## Salida Esperada

Al ejecutar `python notebooks/calibracion.py`, la salida incluye:

1. Información de preprocesamiento (registros cargados, dimensiones)
2. Tabla de barrido de ρ:
```
  rho=0.10 |   1 categories |   0 unassigned |  2 epochs
  rho=0.20 |   3 categories |   0 unassigned |  3 epochs
  ...
  rho=0.90 | 150 categories |   0 unassigned |  5 epochs
```

3. Resumen de la red entrenada:
```
  ART1 (Carpenter-Grossberg) Network Summary
  Input dimension  : 110
  Vigilance (rho)  : 0.50
  Committed cats   : 7
```

4. Tabla cruzada cluster vs. carrera
5. Interpretación de prototipos con bins activos por feature

---

## Parámetros de la Red

### Parámetro de Vigilancia (ρ)

El parámetro más importante. Controla la granularidad del agrupamiento:

| ρ | Efecto |
|---|--------|
| 0.0 - 0.2 | Muy pocas categorías, mezcla perfiles distintos (**sub-clasificación**) |
| 0.3 - 0.5 | Balance moderado, agrupa perfiles similares |
| 0.6 - 0.7 | Categorías más específicas, separación más fina |
| 0.8 - 1.0 | Muchas categorías, cada patrón casi único (**sobre-clasificación**) |

### Parámetro de Elección (L)

Controla la normalización de los pesos bottom-up. Valor estándar: **L = 2.0**. Valores más altos favorecen categorías con prototipos más amplios.

---

## FAQ

### 1. ¿Por qué la red crea demasiadas categorías?

El parámetro de vigilancia (ρ) está demasiado alto. Con ρ cercano a 1.0, la red exige una coincidencia casi perfecta con el prototipo, forzando la creación de nuevas categorías para patrones ligeramente diferentes.

**Solución**: Bajar ρ. Para este dataset, valores entre 0.3 y 0.6 suelen dar resultados equilibrados.

```bash
python CarGross.py -i data/dataset_A.csv -r 0.4
```

### 2. ¿Qué hago si aparecen patrones "unassigned" (-1)?

Esto ocurre si:
- El vector de entrada es todo ceros (vector nulo) — no debería pasar tras el preprocesamiento.
- Se alcanzó el máximo de categorías permitidas.

**Solución**: Aumentar `--max-categories` o verificar que los datos están correctamente binarizados.

### 3. ¿Puedo usar otros datasets que no sean RIASEC?

Sí. CarGross.py acepta cualquier CSV con datos binarios (0/1). Si tu dataset tiene valores continuos, primero debés binarizarlo. Podés adaptar `preprocessor.py` modificando la lista `FEATURES` para tus columnas específicas.

**Requisitos del CSV**:
- Primera fila: headers
- Valores: 0 o 1 (binarios)
- Opcional: última columna como etiqueta (texto)

### 4. ¿Por qué los resultados cambian si reordeno los datos?

ART1 es sensible al orden de presentación. El primer patrón crea la primera categoría, y los siguientes se comparan contra lo ya aprendido. Diferentes órdenes pueden producir clusters distintos pero con coherencia interna similar.

**Mitigación**: El demo usa `seed=42` para reproducibilidad. Podés probar diferentes semillas con `preprocessor.py --seed <N>`.

### 5. ¿Cómo interpreto los prototipos?

Cada prototipo es la intersección (AND) de todos los patrones asignados a ese cluster. Los bins activos indican las características comunes de todos los miembros. Por ejemplo, si el prototipo del cluster 3 tiene `I_score_bin7 = 1` y `Programming_Skill_bin4 = 1`, significa que todos los miembros de ese cluster tienen un score Investigador alto (7+) y habilidad de programación alta (4+).

---

## Referencias

[1] Carpenter, G. A., & Grossberg, S. (1987). "A massively parallel architecture for a self-organizing neural pattern recognition machine." *Computer Vision, Graphics, and Image Processing*, 37(1), 54-115.

[2] Carpenter, G. A., & Grossberg, S. (1987). "ART 2: Self-organization of stable category recognition codes for analog input patterns." *Applied Optics*, 26(23), 4919-4930.

[3] Sánchez-Sinencio, E., & Lau, C. (1992). *Artificial Neural Networks*. IEEE Press. (Material provisto como Lau.pp5.a.11.pdf y Lau.pp12.a.14.pdf)

[4] Dataset RIASEC: https://www.kaggle.com/datasets/svenkateshkumar/student-career-prediction-using-riasec-dataset

[5] Holland, J. L. (1997). *Making Vocational Choices: A Theory of Vocational Personalities and Work Environments*. Psychological Assessment Resources.
