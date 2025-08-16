# Traductor de Señas — MVP (Navegador)

Este proyecto es una aplicación web para el reconocimiento y traducción de lengua de señas en tiempo real, directamente en el navegador. Utiliza la cámara del ordenador para detectar gestos de la mano y los traduce a texto, permitiendo al usuario construir oraciones.

La aplicación es un Prototipo Mínimo Viable (MVP) que demuestra la capacidad de entrenar y utilizar un modelo de inteligencia artificial para el reconocimiento de señas de forma local y privada.

## Descripción del Sistema

El sistema se basa en varias tecnologías web modernas para funcionar de manera autónoma en el navegador del cliente:

- **Detección de Manos:** Utiliza la librería **MediaPipe Hands** de Google para detectar la posición y los 21 puntos de referencia (landmarks) de una o dos manos en el video de la cámara en tiempo real.
- **Normalización de Datos:** Los puntos de referencia de la mano se normalizan para que el reconocimiento sea independiente de la posición, escala y rotación de la mano frente a la cámara.
- **Modelo de IA (TensorFlow.js):**
    - Inicialmente, la aplicación utilizaba un clasificador simple k-NN (k-Nearest Neighbors).
    - La versión actual ha sido mejorada para usar una **Red Neuronal** creada con **TensorFlow.js**.
    - El modelo puede ser entrenado directamente en el navegador por el usuario.
- **Privacidad ("Private by Design"):** Todos los datos, incluyendo los ejemplos de señas capturados, el modelo entrenado y las oraciones guardadas, se almacenan exclusivamente en el `localStorage` del navegador. **Ningún dato personal o de video sale del ordenador del usuario.**
- **Interfaz de Usuario:** La interfaz está diseñada para facilitar el proceso de aprendizaje, entrenamiento y traducción, dividiendo la aplicación en secciones claras.

## Instalación

Al ser una aplicación web estática, no requiere un proceso de instalación complejo. Sin embargo, debido a las políticas de seguridad de los navegadores modernos, no se puede ejecutar simplemente abriendo el archivo `index.html` desde el sistema de archivos (esto impediría la carga del modelo pre-entrenado).

Es necesario servir los archivos desde un servidor web local. La forma más sencilla de hacerlo es con el servidor HTTP que viene incluido en Python.

1.  **Requisitos:** Tener Python 3 instalado en tu sistema.
2.  **Clonar el Repositorio:** Descarga o clona este repositorio en una carpeta local.
3.  **Iniciar el Servidor:**
    - Abre una terminal o línea de comandos.
    - Navega hasta la carpeta donde se encuentran los archivos del proyecto (`index.html`, `script.js`, etc.).
    - Ejecuta el siguiente comando:
      ```bash
      python -m http.server
      ```
4.  **Abrir la Aplicación:**
    - Abre tu navegador web (preferiblemente Chrome, Firefox o Edge).
    - Ve a la siguiente dirección: `http://localhost:8000`
    - La aplicación debería cargarse y estar lista para usar.

## Guía de Uso

La aplicación tiene dos modos principales: "Modo Aprender" y "Modo Traducir".

### 1. Iniciar la Cámara

- Haz clic en el botón **▶️ Iniciar cámara**.
- El navegador te pedirá permiso para acceder a tu cámara. Debes aceptarlo.
- Una vez activa, verás el video de tu cámara en la pantalla.

### 2. Modo Aprender

Este modo te permite enseñarle nuevas señas al sistema.

1.  Haz clic en **🎓 Modo Aprender**.
2.  En el campo de texto "Etiqueta", escribe el nombre de la seña que quieres enseñar (p. ej., `hola`, `adios`, `pregunta`).
3.  Realiza la seña con tu mano frente a la cámara.
4.  Haz clic en el botón **➕ Capturar ejemplo**.
5.  Para una mayor precisión, **repite el paso 4 unas 20-50 veces**, moviendo ligeramente la mano o cambiando el ángulo para que el modelo pueda generalizar mejor.
6.  Repite este proceso para todas las señas que desees enseñar. Necesitas **al menos 2 señas diferentes** para poder entrenar el modelo.

### 3. Entrenar el Modelo

Una vez que has capturado suficientes ejemplos, debes entrenar la red neuronal.

1.  Haz clic en el botón **🧠 Entrenar Modelo**.
2.  El estado del modelo cambiará a "Entrenando..." y verás el progreso del entrenamiento (épocas, pérdida y precisión) en tiempo real.
3.  Espera a que el proceso termine. Cuando finalice, el estado cambiará a "Modelo entrenado y listo".

### 4. Modo Traducir

Con el modelo ya entrenado, puedes empezar a traducir.

1.  Haz clic en **🗣️ Modo Traducir**.
2.  Realiza una de las señas que has entrenado frente a la cámara.
3.  La aplicación mostrará la seña reconocida y su nivel de confianza. Si la confianza supera el umbral definido, la palabra (token) se añadirá a la oración actual.
4.  Usa los botones `␣ Espacio`, `⌫ Borrar`, y `🧹 Limpiar` para construir tu oración.
5.  Cuando termines, puedes guardar la oración en tu lista personal con el botón **💾 Guardar oración**.

### 5. Gestión de Datos y Oraciones

- **Oraciones Guardadas:** Las oraciones que guardas aparecen en una lista. Puedes **editar (✏️)** o **borrar (🗑️)** cada una de ellas.
- **Exportar/Importar:**
    - **💾 Exportar Todo:** Guarda todas tus señas y oraciones en un único archivo JSON (`traductor_senas_datos.json`). Es muy útil para hacer copias de seguridad.
    - **📥 Importar Todo:** Carga un archivo JSON previamente exportado para restaurar tu estado.
- **♻️ Reset total:** Borra permanentemente todos los datos (señas y oraciones) del almacenamiento de tu navegador.

### 6. Atajos de Teclado

- `L`: Activar Modo Aprender.
- `T`: Activar Modo Traducir.
- `Espacio`: Añadir un espacio en la oración.
- `Backspace`: Borrar el último token.
- `.`: Terminar y guardar la oración.
- `,`: Añadir una coma.
