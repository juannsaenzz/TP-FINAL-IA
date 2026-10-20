# Manual de Referencia - Sistema de Test Vocacional (ART1)

## 1. Proceso de Instalación
Para cumplir con el requisito de instalación sencilla, el proyecto utiliza dependencias estándar de Python.
1. Asegúrese de tener Python 3.8 o superior instalado.
2. Abra una terminal en la raíz del proyecto.
3. Instale las dependencias ejecutando:
   `pip install -r requirements.txt`

## 2. Modo de Correr (Test Demo)
Para ejecutar una prueba rápida de demostración con el dataset A y la vigilancia óptima (0.1):
`python src/CarGross.py -i data/dataset_A.csv --rho 0.1`

Para ver además el desglose detallado de los rasgos de prototipo en común que definen cada categoría:
`python src/CarGross.py -i data/dataset_A.csv --rho 0.1 -v`

**Salida Esperada:**
El programa leerá los 100 vectores binarios, inicializará las matrices de la red ART1 y comenzará el proceso de agrupamiento no supervisado. Finalizará imprimiendo por consola la cantidad total de categorías formadas, la cantidad de alumnos por categoría y la distribución vocacional de carreras. Con `-v`, mostrará además los bits específicos de la matriz Top-Down (`W_TD`) que caracterizan al prototipo del grupo.

## 3. Alcances y Limitaciones
**Alcances:**
- El sistema es capaz de aprender categorías nuevas sobre la marcha sin olvidar las anteriores (Dilema de Estabilidad-Plasticidad resuelto).
- Acepta cualquier CSV de entrada siempre y cuando los datos estén binarizados previamente usando codificación dispersa (10 bines por variable).

**Limitaciones:**
- La red ART1 original **solo admite entradas binarias (0 y 1)**. No puede procesar valores continuos sin una fase de preprocesamiento previa.
- Si se ingresa un vector completamente nulo (todo ceros), el sistema arrojará un error, ya que anula matemáticamente la fase de comparación de la red.

## 4. Preguntas Frecuentes (FAQ)

**Q1: ¿Por qué la red genera tantas categorías cuando le paso el Dataset B con rho=0.9?**
**R1:** Porque el parámetro de vigilancia (`rho`) está configurado muy alto. Esto vuelve a la red extremadamente estricta; exige que el vector de entrada sea casi idéntico al prototipo almacenado. Al haber tanta exigencia, la red prefiere crear una categoría nueva antes que agrupar a dos alumnos que difieran mínimamente.

**Q2: Ingresé mi propio CSV pero el programa crashea indicando que no son datos binarios, ¿qué hago?**
**R2:** ART1 no procesa datos continuos. Debe pasar su CSV primero por el script `src/preprocess.py` (o aplicar la misma lógica) para expandir sus variables en un vector de 110 bits (ceros y unos) garantizando una norma mayor a cero.

**Q3: ¿Cómo sabe la red qué carrera recomendar?**
**R3:** En realidad, la red ART1 no "sabe" qué carrera es. Realiza *aprendizaje no supervisado*. Agrupa a los estudiantes estrictamente por la similitud matemática de sus intereses y habilidades. Luego, el especialista humano revisa el clúster formado y, evaluando la etiqueta original, deduce la afinidad profesional del grupo.

---
## REFERENCIAS
[1] Artificial Neural Networks. Edgar Sánchez-Sinencio and Clifford Lau. IEEE Press. 1992. (Material provisto por la cátedra).
