# Universidad Autónoma de Entre Ríos (UADER)
## Facultad de Ciencia y Tecnología (FCyT)
### Instituto de Investigación y Transferencia de Tecnología (IDTI) — Laboratorio de Redes Neuronales

---

# INFORME DE EXPERIMENTACIÓN Y EVALUACIÓN DE RESULTADOS
## Red Neuronal ART1 (Carpenter-Grossberg) aplicada al Agrupamiento No Supervisado de Perfiles Vocacionales RIASEC

- **Materia:** Inteligencia Artificial
- **Ciclo Lectivo:** 2026
- **Cátedra:** Dra. Daniela López De Luise — Alejandro Pérez Museo
- **Grupo:** Grupo 6
- **Integrantes:**
  - Benedetti, Valentino *(Responsable de Arquitectura)*
  - Benedetti, Agustín *(Responsable del Repositorio)*
  - Frachia, Franco *(Responsable del Test)*
  - Nievas, Juan José *(Líder / Documentación)*
  - Sáenz, Juan Ignacio *(Responsable del Código)*
- **Fecha:** Octubre de 2026

---

## 1. Resumen Ejecutivo

El presente trabajo aborda la implementación completa, calibración empírica y evaluación psicopedagógica de una red neuronal **ART1 (Adaptive Resonance Theory 1)**, formulada por Stephen Grossberg y Gail Carpenter (1987). El objetivo consiste en conformar perfiles vocacionales no supervisados a partir del dataset psicométrico RIASEC (2400 registros y 11 dimensiones originales), resolviendo el **dilema de estabilidad-plasticidad**.

Se estructuró un pipeline automatizado en Python (`preprocessor.py`, `CarGross.py`, `run_demo.py`) que:
1. Normaliza y binariza las variables continuas y ordinales en una representación dispersa (*sparse one-hot binning*) de **110 bits**.
2. Divide el conjunto en dos datasets simétricos e independientes de prueba (**Dataset A** y **Dataset B**, con 1200 registros cada uno).
3. Evalúa la dinámica del parámetro de vigilancia $\rho$ mediante un barrido experimental exhaustivo ($\rho \in [0.10, 0.90]$).
4. Analiza la consistencia semántica y psicopedagógica de las categorías emergentes frente a las etiquetas de carrera declaradas.

Los resultados demuestran la convergencia absoluta del algoritmo en **4 épocas**, logrando agrupamientos estables (37 categorías en Dataset A y 42 en Dataset B para $\rho = 0.10$) con un 0% de patrones sin asignar y alta coherencia interna entre aptitudes técnicas y carreras asociadas.

---

## 2. Marco Teórico y Arquitectura ART1

### 2.1 El Dilema de Estabilidad-Plasticidad

En el modelado neuronal clásico, los algoritmos de aprendizaje competitivo sufren de **olvido catastrófico**: la incorporación continua de nueva información altera irreversiblemente los pesos de patrones aprendidos con anterioridad. 

La teoría ART resuelve este conflicto mediante un circuito de retroalimentación resonante y un subsistema de reinicio (*reset*):
- **Plasticidad:** Capacidad de asignar y comprometer neuronas libres en la capa de reconocimiento ($F_2$) cuando el patrón de entrada no guarda suficiente semejanza con ninguna categoría previa.
- **Estabilidad:** Preservación inalterada de los prototipos previamente aprendidos ante entradas que no resuenen con ellos o cuando la categoría ya convergió.

### 2.2 Componentes de la Red ART1 implementada (`CarGross.py`)

```
      Entrada Binaria X (110 bits)
                 |
                 v
        +-----------------+
        |  Capa F1        |<================+
        |  (Comparación)  |                 |
        +-----------------+                 |
           |           ^                    | Pesos Top-Down
Pesos      |           | Intersección       | W_td (Prototipos)
Bottom-Up  |           | S = X AND W_td     |
W_bu       v           |                    |
        +-----------------+                 |
        |  Capa F2        |-----------------+
        | (Reconocimiento)|
        +-----------------+
                 ^
                 | Inhibición (Reset)
        +-----------------+
        |  Mecanismo de   | <--- Similitud vs rho
        |     Reset       |
        +-----------------+
```

1. **Capa F1 (Comparación):** Vector de dimensión $M = 110$. Recibe el patrón binario $X$ y procesa la coincidencia con la categoría seleccionada por $F_2$.
2. **Capa F2 (Reconocimiento):** Conjunto competitivo de neuronas que indexan las categorías aprendidas. Cada neurona activa calcula su función de activación bottom-up:
   $$T_j = \sum_{i=1}^{M} b_{ij} X_i$$
   donde $b_{ij} = \frac{L}{L - 1 + M}$ para nodos no comprometidos, y $b_{ij} = \frac{t_{ji}}{L - 1 + |W^{TD}_j|}$ para nodos comprometidos ($L = 2.0$).
3. **Mecanismo de Reinicio (Reset) y Vigilancia ($\rho$):** Se evalúa la fracción de bits conservados tras el filtrado top-down:
   $$\frac{|X \cap W^{TD}_J|}{|X|} \ge \rho$$
   - Si la condición se cumple: Ocurre **resonancia**. El prototipo top-down se actualiza mediante la operación de conjunción lógica:
     $$W^{TD}_J(t+1) = X \cap W^{TD}_J(t)$$
   - Si la condición no se cumple: Se activa el **reset**, inhibiendo temporalmente el nodo ganador $J$ en $F_2$ durante la presentación del patrón actual y forzando una nueva competencia. Si ninguna categoría existente supera el umbral $\rho$, se recluta un nodo virgen en $F_2$.

---

## 3. Preprocesamiento y Representación del Espacio de Características

El dataset utilizado procede del estudio psicométrico vocacional Kaggle RIASEC (`career_data.csv`), con 2400 registros válidos y 11 variables predictoras continuas/ordinales.

### 3.1 Esquema de Binarización (Sparse One-Hot Binning)

Dado que ART1 opera exclusivamente sobre variables booleanas ($x_i \in \{0, 1\}$), se diseñó una transformación no lineal que preserva el valor métrico sin introducir sesgos de magnitud:

| Variable Original | Rango Original | Transformación / Escala | Binarización (10 bins) |
|-------------------|----------------|-------------------------|------------------------|
| `Math_Score` | 60 – 100 | Dividido por 10 $\rightarrow$ [6.0, 10.0] | `Math_Score_b0` a `Math_Score_b9` |
| `Science_Score` | 60 – 100 | Dividido por 10 $\rightarrow$ [6.0, 10.0] | `Science_Score_b0` a `Science_Score_b9` |
| `Programming_Skill` | 1 – 5 | Mapeo directo a deciles [1, 5] | `Programming_Skill_b0` a `..._b9` |
| `Communication_Skill` | 1 – 5 | Mapeo directo a deciles [1, 5] | `Communication_Skill_b0` a `..._b9` |
| `Logical_Ability` | 1 – 5 | Mapeo directo a deciles [1, 5] | `Logical_Ability_b0` a `..._b9` |
| `R_score` (Realista) | 0 – 10 | Decil directo [0, 10] | `R_score_b0` a `R_score_b9` |
| `I_score` (Investigador) | 0 – 10 | Decil directo [0, 10] | `I_score_b0` a `I_score_b9` |
| `A_score` (Artístico) | 0 – 10 | Decil directo [0, 10] | `A_score_b0` a `A_score_b9` |
| `S_score` (Social) | 0 – 10 | Decil directo [0, 10] | `S_score_b0` a `S_score_b9` |
| `E_score` (Emprendedor) | 0 – 10 | Decil directo [0, 10] | `E_score_b0` a `E_score_b9` |
| `C_score` (Convencional) | 0 – 10 | Decil directo [0, 10] | `C_score_b0` a `C_score_b9` |

**Propiedades del vector resultante:**
- Longitud total: $11 \times 10 = 110$ dimensiones.
- Norma $L_1$ de cada vector de entrada: $|X| = 11$ (exactamente 11 bits activos en 1, uno por cada característica).
- Filtrado de vectores nulos: 0 registros eliminados (completitud 100%).

### 3.2 Partición Experimental
Para cumplir rigurosamente con la consigna de $\ge 2$ datasets de $\ge 50$ registros, el volumen total de 2400 perfiles se dividió estratificadamente en:
- **`data/dataset_A.csv`**: 1200 registros, 110 columnas binarias + columna `Career`.
- **`data/dataset_B.csv`**: 1200 registros, 110 columnas binarias + columna `Career`.

---

## 4. Experimentación 1: Barrido del Parámetro de Vigilancia ($\rho$)

El parámetro de vigilancia $\rho$ modula la tolerancia de la red ante divergencias entre la entrada y el prototipo. Se realizó un barrido experimental exhaustivo sobre el **Dataset A** en el rango $\rho \in [0.10, 0.90]$ con paso de $0.10$:

| $\rho$ | Categorías Formadas | Patrones No Asignados | Épocas de Convergencia | Diagnóstico del Clustering |
|:------:|:-------------------:|:---------------------:|:----------------------:|:---------------------------|
| **0.10** | **37** | **0** | **4** | **Punto Óptimo Operativo** (Equilibrio granularidad/estabilidad) |
| **0.20** | 147 | 0 | 4 | Hiper-especialización incipiente |
| **0.30** | 200 (techo) | 246 | 3 | Saturación de memoria $F_2$; sobre-clasificación severa |
| **0.40** | 200 (techo) | 586 | 3 | Fuerte rechazo por reset; fragmentación extrema |
| **0.50** | 200 (techo) | 749 | 2 | Más del 60% de patrones rechazados |
| **0.60** | 200 (techo) | 817 | 2 | Fallo en resonancia |
| **0.70** | 200 (techo) | 944 | 2 | Prácticamente nula capacidad de generalización |
| **0.80** | 200 (techo) | 993 | 2 | Solo coincidencia casi idéntica aceptada |
| **0.90** | 200 (techo) | 1000 | 2 | Rechazo masivo (colapso por sobre-vigilancia) |

### 4.1 Justificación Matemática del Fenómeno

Un hallazgo empírico fundamental radica en por qué valores aparentemente moderados de vigilancia ($\rho \ge 0.30$) conducen a la saturación rápida de las 200 categorías disponibles en $F_2$.

Dado que cada vector tiene exactamente $|X| = 11$ bits activos:
1. Si un prototipo $W^{TD}$ ya ha incorporado 2 o 3 patrones variados, el operador AND reduce drásticamente sus bits activos (por ejemplo, a 3 o 4 bits).
2. Para que un nuevo estudiante resuene con dicho prototipo, debe compartir esos mismos 3 bits:
   $$\frac{|X \cap W^{TD}|}{|X|} = \frac{3}{11} \approx 0.272$$
3. Por lo tanto, si fijamos $\rho \ge 0.30$, la red exige $|X \cap W^{TD}| \ge \lceil 11 \times 0.30 \rceil = 4$ bits. Si el prototipo solo tiene 3 bits en común, el cociente es $\frac{3}{11} < 0.30$, disparando el reset y obligando a instanciar una nueva categoría en cada paso.

**Conclusión de calibración:** En espacios con binarización one-hot sparse, $\rho = 0.10$ representa el valor natural de vigilancia que permite a ART1 agrupar tendencias generales sin fragmentar el espacio en singleton clusters.

---

## 5. Experimentación 2: Corridas Comparativas con Dataset A y Dataset B

Fijando el parámetro calibrado $\rho = 0.10$ y $L = 2.0$, se ejecutó el entrenamiento completo en ambos datasets.

### 5.1 Cuadro Comparativo Global

| Métrica | Dataset A | Dataset B | Observación |
|---------|:---------:|:---------:|-------------|
| **Patrones de Entrada** | 1200 | 1200 | Sets balanceados e independientes |
| **Dimensión de Entrada ($M$)** | 110 | 110 | 11 variables $\times$ 10 bins |
| **Vigilancia ($\rho$)** | 0.10 | 0.10 | Parámetro óptimo |
| **Límite de Categorías ($F_2$)** | 200 | 200 | Capacidad disponible |
| **Categorías Comprometidas** | **37** | **42** | Variación mínima (13.5%), alta replicabilidad |
| **Patrones No Asignados** | **0 (0.0%)** | **0 (0.0%)** | Cobertura total de la muestra |
| **Épocas hasta Convergencia** | **4** | **4** | Estabilidad rápida garantizada |
| **Reasignaciones en Época 1** | 1200 | 1200 | Fase de asignación inicial |
| **Reasignaciones en Época 2** | 20 | 28 | Refinamiento de frontera |
| **Reasignaciones en Época 3** | 4 | 5 | Estabilización final |
| **Reasignaciones en Época 4** | **0** | **0** | **Resonancia completa y estable** |

---

## 6. Validación Externa y Coherencia Psicopedagógica (Crosstab)

Dado que ART1 es un modelo no supervisado, la red no tuvo acceso a la columna `Career`. Contrastar las categorías emergentes contra los perfiles declarados permite evaluar la validez externa del agrupamiento.

### 6.1 Pureza y Especialización por Carrera

Tanto en Dataset A como en Dataset B surgieron tres tipos diferenciados de clusters:

1. **Clusters de Alta Pureza ($> 70\%$ de homogeneidad):**
   - **Cluster 13 (Dataset A):** 100.0% Entrepreneur.
   - **Cluster 34 (Dataset A):** 100.0% Software Engineer.
   - **Cluster 35 (Dataset A):** 100.0% Doctor.
   - **Cluster 37 (Dataset B):** 100.0% Doctor.
   - **Cluster 39 (Dataset B):** 100.0% Teacher.
   - **Cluster 5 (Dataset A):** 76.9% Entrepreneur.
   - **Cluster 20 (Dataset A):** 72.4% Teacher.

2. **Clusters de Fusión Vocacional Natural:**
   - **Cluster 1 (Dataset A):** 42.6% Software Engineer, 29.6% Data Scientist, 27.8% Accountant. 
     *Explicación psicopedagógica:* Corresponde a perfiles con alto rendimiento matemático y analítico (orientación STEM/Investigador en RIASEC).
   - **Cluster 4 (Dataset A):** 42.5% Doctor, 35.0% Teacher, 22.5% Software Engineer.
     *Explicación psicopedagógica:* Agrupa vocaciones con alta componente social y científica (perfiles S e I de Holland).

3. **Macro-Clusters Nucleares:**
   - **Cluster 0 (Dataset A - 168 patrones):** Distribución uniforme entre Accountant (25.6%), Data Scientist (19.6%), Entrepreneur (17.3%), etc. Actúa como el cluster inicial de entrada que atrae perfiles promedio antes de que se diferencien los subtipos especializados.

### 6.2 Interpretación de Prototipos Aprendidos

Al analizar los vectores de pesos top-down guardados en `data/prototypes_A.csv` y `data/prototypes_B.csv`:
- Los prototipos asociados a **Software Engineer** y **Data Scientist** retienen invariablemente bits activos en los bines superiores de `Logical_Ability` (bines 7 a 9), `Programming_Skill` (bines 6 a 9) y `I_score` (perfil Investigador).
- Los prototipos de **Entrepreneur** consolidan combinaciones de bines altos en `Communication_Skill` y `E_score` (Iniciativa y persuasión).
- Los prototipos de **Doctor** concentran bines elevados en `Science_Score`, `I_score` y `S_score` (interés humanitario y biológico).

Esto demuestra que la red ART1 no agrupa de forma aleatoria, sino que reconstruye las correlaciones latentes del modelo RIASEC de John Holland.

---

## 7. Alcances, Limitaciones y Discusión Técnica

### 7.1 Alcances Demostrados
- **Tiempo de cómputo eficiente:** Convergencia en 4 épocas para 1200 patrones con matrices NumPy puras.
- **Robustez ante datos no vistos:** Dataset B mostró idéntica pauta de estabilización que Dataset A.
- **Auto-documentación e interfaz CLI:** El script `CarGross.py` incluye validación de argumentos, exportación de resultados y manual de ayuda integrado (`--help`).

### 7.2 Limitaciones Intrínsecas
- **Erosión de prototipos por intersección:** La regla de aprendizaje $W^{TD}_{new} = X \cap W^{TD}_{old}$ es monótonamente no creciente (un bit 1 apagado a 0 jamás puede volver a encenderse). En datasets con alta variabilidad intrínseca, esto tiende a adelgazar los prototipos con el paso de los patrones.
- **Sensibilidad al orden de presentación:** Al ser un algoritmo secuencial en línea, el orden de los datos define qué neurona se compromete primero, provocando ligeras variaciones en la cantidad de clusters totales (37 vs 42).

---

## 8. Conclusiones

1. La red neuronal ART1 desarrollada en `CarGross.py` cumplió satisfactoriamente con la totalidad de los requisitos pedagógicos y funcionales del Trabajo Final Integrador.
2. Se resolvió con éxito el dilema de estabilidad-plasticidad: la red no diverge, no sufre olvido catastrófico y alcanza el estado estable en 4 pasadas completas sobre los datos.
3. El preprocesamiento por *one-hot binning* permitió transformar adecuadamente variables métricas a booleanas, revelando que para representaciones dispersas (11 de 110 bits activos), el parámetro de vigilancia debe ser calibrado en la zona baja ($\rho \approx 0.10$) para prevenir la sobre-fragmentación.
4. Las agrupaciones no supervisadas guardan correlación directa con la taxonomía ocupacional RIASEC, confirmando la utilidad del modelo como motor de orientación vocacional automatizado.

---

## 9. Referencias

1. Carpenter, G. A., & Grossberg, S. (1987). *A massively parallel architecture for a self-organizing neural pattern recognition machine*. Computer Vision, Graphics, and Image Processing, 37(1), 54-115.
2. Sánchez-Sinencio, E., & Lau, C. (1992). *Artificial Neural Networks: Paradigms, Applications, and Hardware Implementations*. IEEE Press, pp. 12-14.
3. Grossberg, S. (1988). *Nonlinear neural networks: Principles, mechanisms, and architectures*. Neural Networks, 1(1), 17-61.
4. Holland, J. L. (1997). *Making vocational choices: A theory of vocational personalities and work environments*. Psychological Assessment Resources.
5. Venkateshkumar, S. (2023). *Student Career Prediction Using RIASEC Dataset*. Kaggle Datasets.
