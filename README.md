# Trabajo Final Integrador: Red Neuronal ART1 (Carpenter & Grossberg)
**Materia:** Inteligencia Artificial  
**Institución:** UADER - FCyT / IDTI Lab de Redes Neuronales  

Implementación en Python de la red neuronal ART1 (*Adaptive Resonance Theory 1*) orientada al agrupamiento (*clustering*) vocacional de estudiantes según sus perfiles académicos y de intereses.

---

## Estructura del Repositorio

```text
├── archive/
│   └── career_data.csv          # Dataset original continuo
├── data/
│   ├── dataset_A.csv            # Dataset binarizado A (100 muestras, 110 bits)
│   └── dataset_B.csv            # Dataset binarizado B (100 muestras, 110 bits)
├── notebooks/
│   └── calibracion.py           # Script de calibración y análisis de sensibilidad de rho
├── src/
│   ├── CarGross.py              # Implementación del algoritmo ART1 y CLI interactivo
│   └── preprocess.py            # Binarización y codificación dispersa (110 bits)
├── MANUAL_REFERENCIA.md         # Manual de referencia, alcances, limitaciones y FAQ
├── requirements.txt             # Dependencias del proyecto
└── README.md                    # Documentación principal del repositorio
```

---

## Instalación

1. Clonar el repositorio:
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd "TP FINAL"
   ```

2. Instalar las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

---

## Guía de Ejecución

### 1. Preprocesamiento (Opcional si ya existen los datasets)
Convierte variables continuas en vectores binarios de 110 bits divididos en 10 bines:
```bash
python src/preprocess.py
```

### 2. Calibración del Hiperparámetro de Vigilancia ($\rho$)
Ejecuta el barrido de vigilancia ($\rho \in [0.1, 0.9]$) para evaluar el dilema de estabilidad-plasticidad:
```bash
python notebooks/calibracion.py
```

### 3. Ejecución del Modelo ART1
Ejecución del clasificador con la vigilancia recomendada ($\rho = 0.1$):
```bash
python src/CarGross.py -i data/dataset_A.csv --rho 0.1
```

Ejecución con perfilado de rasgos del prototipo (`--verbose`):
```bash
python src/CarGross.py -i data/dataset_A.csv --rho 0.1 -v
```

O sobre el conjunto de validación:
```bash
python src/CarGross.py -i data/dataset_B.csv --rho 0.1
```

### 4. Modo Manual / Ayuda
Si se ejecuta sin parámetros o con `--help`, despliega la página de manual Unix (`man`):
```bash
python src/CarGross.py --help
```

---

## Documentación Adicional
Para más detalles técnicos sobre la formulación matemática, limitaciones, alcances y preguntas frecuentes, consulte [MANUAL_REFERENCIA.md](MANUAL_REFERENCIA.md).
