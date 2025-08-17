/**
 * @file Sign Language Translator - Main Application Logic
 * @description This file contains all the client-side logic for the sign language translator,
 * including UI management, camera handling, data persistence, and the machine learning model
 * for sign recognition using TensorFlow.js and MediaPipe.
 */

// =================================================================================
// ===== 1. UI UTILITIES & DOM ELEMENTS
// =================================================================================

/**
 * Queries the DOM for a single element.
 * @param {string} s - The CSS selector.
 * @returns {HTMLElement | null} The found element.
 */
const $ = (s) => document.querySelector(s);

/**
 * Queries the DOM for multiple elements.
 * @param {string} s - The CSS selector.
 * @returns {HTMLElement[]} An array of found elements.
 */
const $$ = (s) => Array.from(document.querySelectorAll(s));

/**
 * Logs a message to the UI's log textarea.
 * @param {string} message - The message to log.
 */
const log = (message) => {
  const el = $('#log');
  el.value = `[${new Date().toLocaleTimeString()}] ${message}\n` + el.value;
};

/**
 * Sets the application's status text and dot indicator.
 * @param {string} text - The status message.
 * @param {string} [cls='dot-idle'] - The CSS class for the status dot ('dot-ok', 'dot-warn', 'dot-bad').
 */
const setStatus = (text, cls = 'dot-idle') => {
  $('#statusText').textContent = text;
  const dot = $('#statusDot');
  dot.className = `status-dot ${cls}`;
};

/**
 * Dumps the current application state (samples, sentences, settings) into a textarea for debugging.
 */
const dump = () => {
  const data = { samples, labels: Object.keys(samples), sentences, settings };
  $('#dump').value = JSON.stringify(data, null, 2);
  $('#countSamples').textContent = Object.values(samples).reduce((a, b) => a + b.length, 0);
  $('#labelsList').textContent = Object.keys(samples).join(', ') || '—';
};

// =================================================================================
// ===== 2. GLOBAL STATE
// =================================================================================

// --- Camera & MediaPipe State ---
let videoStream = null;
let camera = null;
let facingMode = 'user';
let running = false;
let drawFlip = true;
let hands = null; // MediaPipe Hands instance
let lastLandmarks = null; // Store the last detected landmarks for capture
let lastFrameTime = performance.now();

// --- Application Logic State ---
let mode = 'translate'; // 'learn' | 'translate'
let paused = false;
let debounce = { label: null, ts: 0 };
let tokens = [];
let sentences = [];
let settings = { debounceMs: 700, minConfidence: 0.6, smoothing: 0.6 };

// --- Data & Model State ---
let samples = {}; // Object to hold captured sign samples, e.g., { 'hola': [[...], [...]] }
let model = null; // The TensorFlow.js model
let labelMap = []; // Maps model output index to a string label, e.g., ['hola', 'adios']
let modelTrained = false; // Flag to track if the model is trained and ready

// =================================================================================
// ===== 3. CORE LOGIC (NORMALIZATION & AI MODEL)
// =================================================================================

/**
 * Normalizes hand landmarks to be independent of position, scale, and rotation.
 * @param {object[]} landmarks - An array of 21 landmark objects from MediaPipe.
 * @returns {number[] | null} A flattened array of 63 normalized coordinates, or null if input is invalid.
 */
function normalizeLandmarks(landmarks) {
  if (!landmarks || !landmarks.length) return null;

  const base = landmarks[0]; // Use wrist as the origin (0,0,0)
  const pts = landmarks.map(p => ({
    x: p.x - base.x,
    y: p.y - base.y,
    z: (p.z || 0) - (base.z || 0)
  }));

  // Calculate a scaling factor based on the distance between wrist and middle finger MCP.
  // This makes the data robust to changes in hand size or distance from the camera.
  const ref = Math.hypot(pts[9].x, pts[9].y, pts[9].z) || 1e-6;

  // Flatten the array of objects into a single array of numbers [x1, y1, z1, x2, y2, z2, ...]
  return pts.flatMap(p => [p.x / ref, p.y / ref, (p.z || 0) / ref]);
}

/**
 * Updates the UI state based on the current application data.
 * Disables/enables buttons and updates status text.
 */
function updateUiState() {
  const numLabels = Object.keys(samples).length;
  const trainBtn = $('#btnTrain');
  const translateBtn = $('#btnTranslate');
  const modelStatusEl = $('#modelStatus');

  if (numLabels < 2) {
    trainBtn.disabled = true;
    modelStatusEl.textContent = `Necesitas >1 etiquetas para entrenar.`;
  } else {
    trainBtn.disabled = false;
    modelStatusEl.textContent = modelTrained ? 'Modelo listo para traducir.' : 'Listo para entrenar.';
  }

  translateBtn.disabled = !modelTrained;
}

/**
 * Creates a new sequential neural network model with TensorFlow.js.
 * @param {number} numClasses - The number of unique sign labels (classes) for the output layer.
 */
function createModel(numClasses) {
  if (numClasses < 2) {
    model = null; // Can't create a model for fewer than 2 classes
    return;
  }
  model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [63], units: 32, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
  model.add(tf.layers.dense({ units: numClasses, activation: 'softmax' }));

  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });
  log(`Modelo TF.js creado con ${numClasses} clases.`);
  $('#modelStatus').textContent = 'Modelo creado, sin entrenar.';
}

/**
 * Trains the TensorFlow.js model on the collected `samples` data.
 */
async function trainModel() {
  modelTrained = false;
  updateUiState();

  labelMap = Object.keys(samples);
  if (labelMap.length < 2) {
    log('Error: se necesitan al menos 2 etiquetas para entrenar.');
    return;
  }

  createModel(labelMap.length);
  if (!model) {
    log('Error: no se pudo crear el modelo.');
    return;
  }

  // --- Prepare data for training ---
  const allSamples = [];
  const allLabels = [];
  for (const label of labelMap) {
    for (const sample of samples[label]) {
      allSamples.push(sample);
      allLabels.push(labelMap.indexOf(label));
    }
  }

  const xs = tf.tensor2d(allSamples); // Input tensor
  const ys = tf.oneHot(tf.tensor1d(allLabels, 'int32'), labelMap.length); // One-hot encoded labels

  log('Iniciando entrenamiento...');
  $('#modelStatus').textContent = 'Entrenando...';
  $('#btnTrain').disabled = true;

  // --- Train the model ---
  await model.fit(xs, ys, {
    epochs: 50,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        const status = `Época: ${epoch + 1}/50, Pérdida: ${logs.loss.toFixed(4)}, Precisión: ${logs.acc.toFixed(4)}`;
        $('#modelStatus').textContent = status;
        log(status);
      }
    }
  });

  log('Entrenamiento completado.');
  modelTrained = true;

  // --- Save the trained model ---
  await model.save('localstorage://sign-language-model');
  localStorage.setItem('mpp_labelMap', JSON.stringify(labelMap));
  log('Modelo guardado en localStorage.');

  updateUiState();

  // --- Clean up tensors ---
  xs.dispose();
  ys.dispose();
}

/**
 * Predicts a sign label from a landmark vector using the trained TF.js model.
 * @param {number[]} vec - The normalized landmark vector (63 elements).
 * @returns {Promise<{label: string|null, conf: number}>} The predicted label and confidence score.
 */
async function predict(vec) {
  if (!model || !modelTrained || labelMap.length === 0) {
    return { label: null, conf: 0 };
  }

  // Create a tensor from the input vector
  const xs = tf.tensor2d([vec]);

  // Predict
  const prediction = model.predict(xs);
  const probabilities = await prediction.data();

  // Find the index with the highest probability
  let maxProb = 0;
  let maxIndex = -1;
  for (let i = 0; i < probabilities.length; i++) {
    if (probabilities[i] > maxProb) {
      maxProb = probabilities[i];
      maxIndex = i;
    }
  }

  // Clean up tensors to prevent memory leaks
  xs.dispose();
  prediction.dispose();

  if (maxIndex !== -1) {
    return { label: labelMap[maxIndex], conf: maxProb };
  } else {
    return { label: null, conf: 0 };
  }
}

// =================================================================================
// ===== 4. UI & APPLICATION FLOW
// =================================================================================

/** Renders the current tokens into the sentence construction area. */
function renderTokens() {
  const el = $('#tokens');
  el.innerHTML = '';
  tokens.forEach(t => {
    const b = document.createElement('div');
    b.className = 'chip';
    b.textContent = t;
    el.appendChild(b);
  });
  $('#sentence').textContent = tokens.join(' ');
}

/** Adds a new token to the current sentence. */
function addToken(t) {
  if (!t) return;
  tokens.push(t);
  renderTokens();
}

/** Renders the list of saved sentences with edit and delete buttons. */
function renderSavedSentences() {
  const listEl = $('#savedSentencesList');
  listEl.innerHTML = '';
  if (!sentences || sentences.length === 0) {
    listEl.innerHTML = '<div class="list-group-item text-body-secondary">No hay oraciones guardadas.</div>';
    return;
  }
  sentences.forEach(s => {
    const row = document.createElement('div');
    row.className = 'list-group-item d-flex justify-content-between align-items-center';

    const text = document.createElement('span');
    text.textContent = s.text;

    const actions = document.createElement('div');
    const editBtn = document.createElement('button');
    editBtn.className = 'btn btn-sm btn-outline-secondary';
    editBtn.innerHTML = '✏️';
    editBtn.dataset.action = 'edit';
    editBtn.dataset.timestamp = s.ts;
    const delBtn = document.createElement('button');
    delBtn.className = 'btn btn-sm btn-outline-danger ms-2';
    delBtn.innerHTML = '🗑️';
    delBtn.dataset.action = 'delete';
    delBtn.dataset.timestamp = s.ts;
    actions.appendChild(editBtn);
    actions.appendChild(delBtn);

    row.appendChild(text);
    row.appendChild(actions);
    listEl.appendChild(row);
  });
}

// =================================================================================
// ===== 5. DATA PERSISTENCE (SAVE & LOAD)
// =================================================================================

/** Saves samples and sentences to localStorage. */
function save() {
  localStorage.setItem('mpp_samples', JSON.stringify(samples));
  localStorage.setItem('mpp_sentences', JSON.stringify(sentences));
  dump();
}

/**
 * Loads application data on startup.
 * It first tries to load a trained model from localStorage.
 * If that fails, it loads sample data from localStorage or a default JSON file.
 */
async function load() {
  // Load sample and sentence data first
  const savedSamples = localStorage.getItem('mpp_samples');
  if (savedSamples && savedSamples !== '{}') {
    samples = JSON.parse(savedSamples);
    sentences = JSON.parse(localStorage.getItem('mpp_sentences') || '[]');
    dump();
  } else {
    try {
      const response = await fetch('modelo_inicial.json');
      if (!response.ok) throw new Error('Network response was not ok.');
      const data = await response.json();
      if (data && data.samples) {
        samples = data.samples;
        sentences = data.sentences || [];
        settings = data.settings || settings;
        save();
        log('Modelo pre-entrenado cargado.');
      }
    } catch (err) {
      console.error('Error al cargar el modelo inicial:', err);
      log('Error fatal: no se pudo cargar el modelo inicial.');
    }
  }

  // After loading samples, try to load the trained model
  try {
    model = await tf.loadLayersModel('localstorage://sign-language-model');
    labelMap = JSON.parse(localStorage.getItem('mpp_labelMap') || '[]');
    modelTrained = true;
    log('Modelo de IA cargado desde localStorage.');
  } catch (error) {
    log('No se encontró un modelo de IA guardado. Es necesario entrenar.');
    modelTrained = false;
    createModel(Object.keys(samples).length);
  }
}

// =================================================================================
// ===== 6. CAMERA & MEDIAPIPE INTEGRATION
// =================================================================================

const videoEl = $('#video');
const canvasEl = $('#overlay');
const ctx = canvasEl.getContext('2d');

/** Initializes and starts the camera stream and MediaPipe processing. */
async function startCamera() {
  try {
    const sel = $('#selCam');
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cams = devices.filter(d => d.kind === 'videoinput');
    sel.innerHTML = cams.map((c, i) => `<option value="${c.deviceId}">${c.label || 'Cámara ' + (i + 1)}</option>`).join('');
    const constraints = { video: sel.value ? { deviceId: { exact: sel.value } } : { facingMode } };
    videoStream = await navigator.mediaDevices.getUserMedia(constraints);
    videoEl.srcObject = videoStream;
    await videoEl.play();
    videoEl.style.visibility = 'hidden';

    canvasEl.width = videoEl.videoWidth;
    canvasEl.height = videoEl.videoHeight;

    hands = new Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });
    hands.setOptions({ maxNumHands: 2, modelComplexity: 1, minDetectionConfidence: 0.6, minTrackingConfidence: 0.6 });
    hands.onResults(onResults);

    const camUtils = new Camera(videoEl, {
      onFrame: async () => { await hands.send({ image: videoEl }); },
      width: videoEl.videoWidth,
      height: videoEl.videoHeight
    });
    camUtils.start();

    running = true;
    setStatus('capturando', 'dot-ok');
    log('Cámara iniciada');
    loopFPS();
  } catch (e) {
    log('Error cámara: ' + e.message);
    setStatus('error', 'dot-bad');
  }
}

/** Stops the camera stream. */
function stopCamera() {
  try {
    videoStream?.getTracks()?.forEach(t => t.stop());
    running = false;
    setStatus('inactivo', 'dot-idle');
    videoEl.style.visibility = 'visible';
  } catch (e) {
    console.error(e);
  }
}

/**
 * Callback function for MediaPipe Hands. Processes detection results.
 * @param {object} results - The detection results from MediaPipe.
 */
async function onResults(results) {
  lastLandmarks = (results.multiHandLandmarks && results.multiHandLandmarks[0]) ? results.multiHandLandmarks[0] : null;

  ctx.save();
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  if (drawFlip) {
    ctx.translate(canvasEl.width, 0);
    ctx.scale(-1, 1);
  }
  if (results.image) {
    ctx.drawImage(results.image, 0, 0, canvasEl.width, canvasEl.height);
  }

  if (results.multiHandLandmarks && !paused) {
    for (const landmarks of results.multiHandLandmarks) {
      drawConnectors(ctx, landmarks, HAND_CONNECTIONS, { color: '#6ae3ff', lineWidth: 2 });
      drawLandmarks(ctx, landmarks, { color: '#ffffff', lineWidth: 1 });

      const vec = normalizeLandmarks(landmarks);
      if (vec && mode === 'translate') {
        const { label, conf } = await predict(vec);
        $('#conf').textContent = conf.toFixed(2);
        if (label && conf > settings.minConfidence) {
          const now = Date.now();
          if (debounce.label !== label || (now - debounce.ts) > settings.debounceMs) {
            debounce = { label, ts: now };
            $('#lastLabel').textContent = label;
            addToken(label);
          }
        }
      }
    }
  }
  ctx.restore();
}

/** Main loop to calculate and display frames per second. */
function loopFPS() {
  if (!running) return;
  const now = performance.now();
  const dt = now - lastFrameTime;
  lastFrameTime = now;
  const fps = Math.round(1000 / dt);
  $('#fps').textContent = isFinite(fps) ? fps : 0;
  requestAnimationFrame(loopFPS);
}

// =================================================================================
// ===== 7. EVENT LISTENERS
// =================================================================================

// --- Camera Controls ---
$('#btnStart').onclick = startCamera;
$('#btnStop').onclick = stopCamera;
$('#btnFlip').onclick = () => { drawFlip = !drawFlip; };
$('#selCam').onchange = () => { stopCamera(); startCamera(); };

// --- Mode & Recognition Controls ---
$('#btnLearn').onclick = (e) => { mode = 'learn'; e.currentTarget.setAttribute('aria-pressed', 'true'); $('#btnTranslate').setAttribute('aria-pressed', 'false'); log('Modo Aprender'); };
$('#btnTranslate').onclick = (e) => { mode = 'translate'; e.currentTarget.setAttribute('aria-pressed', 'true'); $('#btnLearn').setAttribute('aria-pressed', 'false'); log('Modo Traducir'); };
$('#btnHold').onclick = () => { paused = !paused; $('#btnHold').textContent = paused ? '▶️ Reanudar' : '⏸️ Pausa'; };
$('#btnTrain').onclick = trainModel;

// --- Data Capture & Management ---
$('#btnCapture').onclick = () => {
  const label = $('#labelInput').value.trim().toLowerCase();
  if (!label) {
    alert('Pon una etiqueta');
    return;
  }
  if (!lastLandmarks) {
    alert('No hay mano detectada');
    log('Intento de captura fallido: No se detectaron landmarks.');
    return;
  }

  log(`Capturando para la etiqueta: "${label}"...`);
  const vec = normalizeLandmarks(lastLandmarks);
  if (!vec) {
      log('Error: No se pudieron normalizar los landmarks.');
      return;
  }

  samples[label] = samples[label] || [];
  samples[label].push(vec);

  $('#lastLabel').textContent = label;
  log(`Muestra capturada para "${label}". Total para esta etiqueta: ${samples[label].length}.`);

  save();
  log('Llamando a save() para persistir los datos.');

  modelTrained = false;
  updateUiState();
};

$('#btnClearLabel').onclick = async () => {
  const label = $('#labelInput').value.trim().toLowerCase();
  if (!label) return;
  if (samples[label]) {
    delete samples[label];
    log(`Eliminada etiqueta "${label}"`);
    save();
    try {
      await tf.models.remove('localstorage://sign-language-model');
      localStorage.removeItem('mpp_labelMap');
      log('Modelo de IA eliminado.');
    } catch (error) { /* Ignorar si no existía */ }
    modelTrained = false;
    updateUiState();
  }
};

$('#btnExportAll').onclick = () => {
  const data = { samples, sentences, settings };
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'traductor_senas_datos.json';
  a.click();
  log('Datos exportados');
};

$('#btnImportAll').onclick = () => $('#fileImportAll').click();

$('#fileImportAll').onchange = (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const r = new FileReader();
  r.onload = () => {
    try {
      const data = JSON.parse(r.result);
      if (data && data.samples && data.sentences && data.settings) {
        samples = data.samples;
        sentences = data.sentences;
        settings = data.settings;
        save();
        renderTokens();
        renderSavedSentences();
        modelTrained = false;
        updateUiState();
        log('Datos importados correctamente');
      } else {
        alert('El archivo de datos no tiene el formato esperado.');
        log('Error: el archivo de importación no es válido');
      }
    } catch (err) {
      alert('Error al leer el archivo JSON: ' + err.message);
      log('Error de importación: ' + err.message);
    }
  };
  r.readAsText(f);
};

$('#btnReset').onclick = async () => {
  if (confirm('¿Borrar todo?')) {
    samples = {};
    sentences = [];
    tokens = [];
    modelTrained = false;
    save();
    renderTokens();
    renderSavedSentences();
    try {
      await tf.models.remove('localstorage://sign-language-model');
      localStorage.removeItem('mpp_labelMap');
      log('Modelo de IA eliminado.');
    } catch (error) { /* Ignorar si no existía */ }
    log('Reset completo');
    updateUiState();
  }
};

// --- Sentence Controls ---
$('#btnSpace').onclick = () => { tokens.push(''); renderTokens(); };
$('#btnDelete').onclick = () => { tokens.pop(); renderTokens(); };
$('#btnClear').onclick = () => { tokens = []; renderTokens(); };
$('#btnFinish').onclick = () => { if (tokens.length) { const s = tokens.join(' ').replace(/\s+/g, ' ').trim() + '.'; sentences.push({ text: s, ts: Date.now() }); tokens = []; renderTokens(); save(); renderSavedSentences(); log('Oración guardada: ' + s); } };
$('#btnComma').onclick = () => { tokens.push(','); renderTokens(); };
$('#btnSaveSentence').onclick = () => { const s = $('#sentence').textContent.trim(); if (!s) return; sentences.push({ text: s, ts: Date.now() }); save(); renderSavedSentences(); log('Oración guardada (manual)'); };

// --- Saved Sentences List Handler (Event Delegation) ---
$('#savedSentencesList').onclick = (e) => {
  const target = e.target.closest('button');
  if (!target) return;

  const action = target.dataset.action;
  const timestamp = parseInt(target.dataset.timestamp, 10);
  const sentenceIndex = sentences.findIndex(s => s.ts === timestamp);

  if (sentenceIndex === -1) return;

  if (action === 'delete') {
    if (confirm('¿Seguro que quieres borrar esta oración?')) {
      sentences.splice(sentenceIndex, 1);
      save();
      renderSavedSentences();
      log('Oración borrada.');
    }
  } else if (action === 'edit') {
    const currentText = sentences[sentenceIndex].text;
    const newText = prompt('Edita la oración:', currentText);
    if (newText && newText.trim() !== currentText) {
      sentences[sentenceIndex].text = newText.trim();
      save();
      renderSavedSentences();
      log('Oración actualizada.');
    }
  }
};

// --- Keyboard Shortcuts ---
document.addEventListener('keydown', (ev) => {
  if (ev.key === ' ') { ev.preventDefault(); $('#btnSpace').click(); }
  if (ev.key === 'Backspace') { $('#btnDelete').click(); }
  if (ev.key === 'l' || ev.key === 'L') { $('#btnLearn').click(); }
  if (ev.key === 't' || ev.key === 'T') { $('#btnTranslate').click(); }
  if (ev.key === ',') { $('#btnComma').click(); }
  if (ev.key === '.') { $('#btnFinish').click(); }
});

// =================================================================================
// ===== 8. INITIALIZATION
// =================================================================================

(async () => {
  await load();
  renderTokens();
  renderSavedSentences();
  setStatus('inactivo', 'dot-idle');
  updateUiState();
})();

// --- Theme Switcher Logic ---
const themeToggleBtn = document.getElementById('theme-toggle-btn');
const htmlEl = document.documentElement;

const applyTheme = (theme) => {
  htmlEl.setAttribute('data-bs-theme', theme);
  localStorage.setItem('theme', theme);
};

const savedTheme = localStorage.getItem('theme') || 'light';
applyTheme(savedTheme);

themeToggleBtn.addEventListener('click', () => {
  const currentTheme = htmlEl.getAttribute('data-bs-theme');
  const newTheme = currentTheme === 'light' ? 'dark' : 'light';
  applyTheme(newTheme);
});
