
# Clasificador de Texto Federado con LoRA y Flower

Este proyecto demuestra un sistema de aprendizaje federado para la clasificación de texto utilizando Flower, Hugging Face Transformers y LoRA (Low-Rank Adaptation) para un entrenamiento eficiente.

## Características Principales

- **Aprendizaje Federado**: Entrena un modelo en datos distribuidos sin centralizarlos, mejorando la privacidad.
- **Eficiencia con LoRA**: Utiliza LoRA para adaptar un modelo pre-entrenado (distilbert-base-multilingual-cased) con menos parámetros entrenables, reduciendo la sobrecarga computacional.
- **Clasificación de Texto**: El modelo está diseñado para clasificar textos como "sensibles" o "no sensibles".
- **Componentes Modulares**: Código claramente separado para el servidor, cliente, configuración, manejo de datos e inferencia.
- **Simulación y Datos Reales**: Capacidad para ejecutarse con un conjunto de datos de juguete generado o cargar datos desde un archivo CSV.

---

## Arquitectura de Aprendizaje Federado

El sistema sigue una arquitectura cliente-servidor gestionada por Flower. El servidor central coordina el proceso de entrenamiento, mientras que los clientes entrenan el modelo en sus datos locales.

```ascii
          +-------------------+
          |   Servidor FL   |
          |  (Flower)       |
          +--------+--------+
                   |
     +-------------+-------------+
     |             |             |
+----v----+   +----v----+   +----v----+
| Cliente |   | Cliente |   | Cliente |
|   (1)   |   |   (2)   |   |   (n)   |
+---------+   +---------+   +---------+
```

1.  **Inicialización**: El servidor inicializa el modelo global.
2.  **Distribución**: El servidor envía los parámetros del modelo a los clientes.
3.  **Entrenamiento Local**: Cada cliente entrena el modelo en sus datos locales utilizando LoRA.
4.  **Agregación**: Los clientes envían sus parámetros actualizados al servidor, que los agrega para mejorar el modelo global.
5.  **Repetición**: El proceso se repite durante varias rondas.

---

## Flujo de Ejecución

Para ejecutar el sistema, necesitas iniciar el servidor y luego uno o más clientes. Finalmente, puedes usar el script de inferencia para probar el modelo.

```ascii
+------------------+      +------------------+      +------------------+
| 1. Iniciar       |      | 2. Iniciar       |      | 3. Ejecutar      |
|    Servidor      | ---> |    Clientes      | ---> |    Inferencia    |
|                  |      |                  |      |                  |
| `python -m src.server` | | `python -m src.client` | | `python -m src.inference`|
+------------------+      +------------------+      +------------------+
```

---

## Ejemplo de Ejecución

Sigue estos pasos para poner en marcha el sistema.

### 1. Instalar Dependencias

Asegúrate de tener todas las dependencias instaladas:

```bash
pip install -r requirements.txt
```

### 2. Iniciar el Servidor

Abre una terminal y ejecuta el siguiente comando para iniciar el servidor de Flower:

```bash
python -m src.server
```

El servidor esperará a que se conecten los clientes.

### 3. Iniciar los Clientes

Abre **nuevas terminales** para cada cliente que quieras iniciar. El sistema está configurado para esperar al menos 2 clientes (`MIN_AVAILABLE_CLIENTS = 2`).

**Terminal del Cliente 1:**

```bash
python -m src.client --client-id 1
```

**Terminal del Cliente 2:**

```bash
python -m src.client --client-id 2
```

> **Nota**: Si tienes datos locales, puedes pasarlos con el argumento `--data-path mi_archivo.csv`. Si no, los clientes usarán un conjunto de datos de juguete.

Los clientes se conectarán al servidor y comenzará el entrenamiento federado.

### 4. Ejecutar Inferencia

Una vez que el entrenamiento haya finalizado (o incluso mientras se ejecuta, para probar), puedes usar el script de inferencia para clasificar texto.

Abre otra terminal y ejecuta:

```bash
python -m src.inference --text "Este es un documento privado con mi contraseña"
```

El script cargará el modelo y predecirá si el texto es sensible.

**Salida de Ejemplo:**

```
------------------------------
Text: Este es un documento privado con mi contraseña
Sensitive Probability: 0.9876
Prediction: SENSITIVE
------------------------------
```

---

## Configuración

Puedes modificar el comportamiento del sistema a través del archivo `src/config.py`. Algunas de las opciones clave son:

- **MODEL_NAME**: El modelo base de Hugging Face a utilizar.
- **LORA_R**, **LORA_ALPHA**: Parámetros de configuración de LoRA.
- **SERVER_ADDRESS**: La dirección IP y el puerto del servidor.
- **NUM_ROUNDS**: El número de rondas de entrenamiento federado.
- **MIN_FIT_CLIENTS**, **MIN_AVAILABLE_CLIENTS**: El número mínimo de clientes para el entrenamiento.
- **LOCAL_EPOCHS**: El número de épocas que cada cliente entrena localmente en cada ronda.
- **BATCH_SIZE**, **LEARNING_RATE**: Hiperparámetros de entrenamiento.
