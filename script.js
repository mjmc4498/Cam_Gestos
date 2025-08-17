// ===== Utilidades UI =====
const $ = (s)=>document.querySelector(s);
const $$ = (s)=>Array.from(document.querySelectorAll(s));
const log = (m)=>{ const el=$('#log'); el.value = `[${new Date().toLocaleTimeString()}] ${m}\n` + el.value; }
const setStatus = (txt,cls='dot-idle')=>{ $('#statusText').textContent = txt; const dot=$('#statusDot'); dot.className = `status-dot ${cls}` }
const dump = ()=>{ const data={samples,labels:Object.keys(samples),sentences,settings}; $('#dump').value = JSON.stringify(data,null,2); $('#countSamples').textContent = Object.values(samples).reduce((a,b)=>a+b.length,0); $('#labelsList').textContent = Object.keys(samples).join(', ')||'—'; }

// ===== Estado global =====
let videoStream=null, camera=null, facingMode='user', running=false, drawFlip=true;
let hands=null; // MediaPipe Hands
let lastLandmarks = null;
let lastFrameTime=performance.now();
let mode='translate'; // 'learn' | 'translate'
let paused=false;
let debounce={label:null, ts:0};
let tokens=[]; let sentences=[];
let settings={debounceMs:700, minConfidence:0.6, smoothing:0.6};
let samples = JSON.parse(localStorage.getItem('mpp_samples')||'{}');

// ===== Normalización de landmarks =====
function normalizeLandmarks(landmarks){
  // landmarks: [{x,y,z} * 21] con coords relativas al frame [0..1]
  if(!landmarks||!landmarks.length) return null;
  const base = landmarks[0]; // wrist
  const pts = landmarks.map(p=>({x:p.x-base.x, y:p.y-base.y, z:(p.z||0)-(base.z||0)}));
  // escala por distancia muñeca→medio(9)
  const ref = Math.hypot(pts[9].x, pts[9].y, pts[9].z)||1e-6;
  return pts.flatMap(p=>[p.x/ref, p.y/ref, (p.z||0)/ref]);
}

// ===== Modelo de IA con TensorFlow.js =====
let model = null;
let labelMap = [];
let modelTrained = false;

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

function createModel(numClasses) {
  if (numClasses < 2) {
    // No se puede crear un modelo para una sola clase.
    // Se podría manejar este caso, pero por ahora lo dejamos así.
    model = null;
    return;
  }
  model = tf.sequential();
  model.add(tf.layers.dense({inputShape: [63], units: 32, activation: 'relu'}));
  model.add(tf.layers.dense({units: 16, activation: 'relu'}));
  model.add(tf.layers.dense({units: numClasses, activation: 'softmax'}));

  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });
  log(`Modelo TF.js creado con ${numClasses} clases.`);
  $('#modelStatus').textContent = 'Modelo creado, sin entrenar.';
}

async function trainModel() {
  modelTrained = false;
  updateUiState();

  labelMap = Object.keys(samples);
  if (labelMap.length < 2) {
    log('Error: se necesitan al menos 2 etiquetas para entrenar.');
    // No need for alert, UI state handles it.
    return;
  }

  createModel(labelMap.length);

  if (!model) {
    log('Error: no se pudo crear el modelo.');
    return;
  }

  // Preparar datos para entrenamiento
  const allSamples = [];
  const allLabels = [];
  for (const label of labelMap) {
    for (const sample of samples[label]) {
      allSamples.push(sample);
      allLabels.push(labelMap.indexOf(label));
    }
  }

  const xs = tf.tensor2d(allSamples);
  const ys = tf.oneHot(tf.tensor1d(allLabels, 'int32'), labelMap.length);

  log('Iniciando entrenamiento...');
  $('#modelStatus').textContent = 'Entrenando...';
  $('#btnTrain').disabled = true; // Disable button during training

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
  updateUiState();

  // Limpiar tensores
  xs.dispose();
  ys.dispose();
}

async function predict(vec) {
  if (!model || labelMap.length === 0) {
    return {label: null, conf: 0};
  }

  // Crear tensor desde el vector de entrada
  const xs = tf.tensor2d([vec]);

  // Realizar la predicción
  const prediction = model.predict(xs);
  const probabilities = await prediction.data();

  // Encontrar el índice con la probabilidad más alta
  let maxProb = 0;
  let maxIndex = -1;
  for (let i = 0; i < probabilities.length; i++) {
    if (probabilities[i] > maxProb) {
      maxProb = probabilities[i];
      maxIndex = i;
    }
  }

  // Limpiar tensores
  xs.dispose();
  prediction.dispose();

  if (maxIndex !== -1) {
    return {label: labelMap[maxIndex], conf: maxProb};
  } else {
    return {label: null, conf: 0};
  }
}



// ===== Oraciones =====
function renderTokens(){ const el=$('#tokens'); el.innerHTML=''; tokens.forEach(t=>{ const b=document.createElement('div'); b.className='chip'; b.textContent=t; el.appendChild(b); }); $('#sentence').textContent = tokens.join(' '); }
function addToken(t){ if(!t) return; tokens.push(t); renderTokens(); }

// ===== Persistencia =====
function save(){ localStorage.setItem('mpp_samples', JSON.stringify(samples)); localStorage.setItem('mpp_sentences', JSON.stringify(sentences)); dump(); }
function load() {
  const savedSamples = localStorage.getItem('mpp_samples');
  if (savedSamples && savedSamples !== '{}') {
    samples = JSON.parse(savedSamples);
    sentences = JSON.parse(localStorage.getItem('mpp_sentences') || '[]');
    dump();
    log('Datos de usuario cargados desde localStorage');
    createModel(Object.keys(samples).length);
  } else {
    // Si no hay datos, cargar el modelo inicial
    fetch('modelo_inicial.json')
      .then(response => {
        if (!response.ok) {
          throw new Error('No se pudo cargar el modelo inicial: ' + response.statusText);
        }
        return response.json();
      })
      .then(data => {
        if (data && data.samples && data.sentences && data.settings) {
          samples = data.samples;
          sentences = data.sentences;
          settings = data.settings;
          save();
          log('Modelo pre-entrenado cargado');
          createModel(Object.keys(samples).length);
        } else {
          log('Error: el modelo inicial no es válido');
        }
      })
      .catch(err => {
        console.error('Error al cargar el modelo inicial:', err);
        log('Error fatal: no se pudo cargar el modelo inicial.');
      });
  }
}

// ===== Cámara + MediaPipe Hands =====
const videoEl = $('#video');
const canvasEl = $('#overlay');
const ctx = canvasEl.getContext('2d');

async function startCamera(){
  try{
    const sel = $('#selCam');
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cams = devices.filter(d=>d.kind==='videoinput');
    sel.innerHTML = cams.map((c,i)=>`<option value="${c.deviceId}">${c.label||'Cámara '+(i+1)}</option>`).join('');
    const constraints = {video: sel.value?{deviceId:{exact:sel.value}}:{facingMode}};
    videoStream = await navigator.mediaDevices.getUserMedia(constraints);
    videoEl.srcObject = videoStream; await videoEl.play();
    canvasEl.width = videoEl.videoWidth; canvasEl.height = videoEl.videoHeight;
    // MediaPipe Hands
    hands = new Hands({locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`});
    hands.setOptions({ maxNumHands: 2, modelComplexity: 1, minDetectionConfidence: 0.6, minTrackingConfidence: 0.6 });
    hands.onResults(onResults);
    const camUtils = new Camera(videoEl, { onFrame: async () => { await hands.send({ image: videoEl }); }, width: videoEl.videoWidth, height: videoEl.videoHeight });
    camUtils.start();
    running=true; setStatus('capturando','dot-ok'); log('Cámara iniciada');
    loopFPS();
  }catch(e){ log('Error cámara: '+e.message); setStatus('error','dot-bad'); }
}

function stopCamera(){ try{ videoStream?.getTracks()?.forEach(t=>t.stop()); running=false; setStatus('inactivo','dot-idle'); }catch{ }
}

async function onResults(results){
  lastLandmarks = (results.multiHandLandmarks && results.multiHandLandmarks[0]) ? results.multiHandLandmarks[0] : null;

  // Dibujo
  ctx.save(); ctx.clearRect(0,0,canvasEl.width,canvasEl.height);
  if(drawFlip){ ctx.translate(canvasEl.width,0); ctx.scale(-1,1); }
  if(results.image) ctx.drawImage(results.image,0,0,canvasEl.width,canvasEl.height);

  if(results.multiHandLandmarks && !paused){
    for (const landmarks of results.multiHandLandmarks) {
      // dibuja esqueletos
      drawConnectors(ctx, landmarks, HAND_CONNECTIONS, {color:'#6ae3ff', lineWidth:2});
      drawLandmarks(ctx, landmarks, {color:'#ffffff', lineWidth:1});
      const vec = normalizeLandmarks(landmarks);
      if(vec){
        if(mode==='learn'){
          // sólo dibujo; la captura es manual con botón
        } else if(mode==='translate'){
          const {label, conf} = await predict(vec);
          $('#conf').textContent = conf.toFixed(2);
          if(label && conf > settings.minConfidence){
            const now = Date.now();
            if(debounce.label!==label || (now - debounce.ts) > settings.debounceMs){
              debounce={label,ts:now};
              $('#lastLabel').textContent = label;
              addToken(label);
            }
          }
        }
      }
    }
  }
  ctx.restore();
}

function loopFPS(){ if(!running) return; const now=performance.now(); const dt=now-lastFrameTime; lastFrameTime=now; const fps=Math.round(1000/dt); $('#fps').textContent = isFinite(fps)?fps:0; requestAnimationFrame(loopFPS); }

// ===== Eventos UI =====
function renderSavedSentences() {
  const listEl = $('#savedSentencesList');
  listEl.innerHTML = '';
  if (!sentences || sentences.length === 0) {
    listEl.innerHTML = '<div class="muted">No hay oraciones guardadas.</div>';
    return;
  }
  sentences.forEach(s => {
    const row = document.createElement('div');
    row.style.display = 'flex';
    row.style.justifyContent = 'space-between';
    row.style.alignItems = 'center';
    row.style.gap = '8px';

    const text = document.createElement('span');
    text.textContent = s.text;
    text.style.flex = 1;

    const actions = document.createElement('div');
    const editBtn = document.createElement('button');
    editBtn.textContent = '✏️';
    editBtn.dataset.action = 'edit';
    editBtn.dataset.timestamp = s.ts;
    const delBtn = document.createElement('button');
    delBtn.textContent = '🗑️';
    delBtn.dataset.action = 'delete';
    delBtn.dataset.timestamp = s.ts;
    actions.appendChild(editBtn);
    actions.appendChild(delBtn);

    row.appendChild(text);
    row.appendChild(actions);
    listEl.appendChild(row);
  });
}

$('#btnStart').onclick = startCamera;
$('#btnStop').onclick = stopCamera;
$('#btnFlip').onclick = ()=>{ drawFlip=!drawFlip };
$('#selCam').onchange = ()=>{ stopCamera(); startCamera(); };

$('#btnLearn').onclick = (e)=>{ mode='learn'; e.currentTarget.setAttribute('aria-pressed','true'); $('#btnTranslate').setAttribute('aria-pressed','false'); log('Modo Aprender'); };
$('#btnTranslate').onclick = (e)=>{ mode='translate'; e.currentTarget.setAttribute('aria-pressed','true'); $('#btnLearn').setAttribute('aria-pressed','false'); log('Modo Traducir'); };
$('#btnHold').onclick = ()=>{ paused=!paused; $('#btnHold').textContent = paused?'▶️ Reanudar':'⏸️ Pausa'; };
$('#btnTrain').onclick = trainModel;


$('#btnCapture').onclick = ()=>{
  const label = $('#labelInput').value.trim().toLowerCase(); if(!label){ alert('Pon una etiqueta'); return; }
  if(!lastLandmarks){ alert('No hay mano detectada'); return; }
  const vec = normalizeLandmarks(lastLandmarks);
  samples[label] = samples[label]||[]; samples[label].push(vec);
  $('#lastLabel').textContent = label;
  log(`Capturada muestra para "${label}" (#${samples[label].length})`);
  save();
  modelTrained = false; // El modelo necesita re-entrenamiento
  updateUiState();
};

$('#btnClearLabel').onclick = ()=>{
  const label = $('#labelInput').value.trim().toLowerCase(); if(!label) return;
  if(samples[label]){
    delete samples[label];
    log(`Eliminada etiqueta "${label}"`);
    save();
    modelTrained = false; // El modelo necesita re-entrenamiento
    updateUiState();
  }
};

$('#btnExportAll').onclick = ()=>{
  const data = { samples, sentences, settings };
  const blob = new Blob([JSON.stringify(data, null, 2)],{type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'traductor_senas_datos.json';
  a.click();
  log('Datos exportados');
};
$('#btnImportAll').onclick = ()=> $('#fileImportAll').click();
$('#fileImportAll').onchange = (e)=>{
  const f = e.target.files[0];
  if(!f) return;
  const r = new FileReader();
  r.onload = ()=>{
    try {
      const data = JSON.parse(r.result);
      if (data && data.samples && data.sentences && data.settings) {
        samples = data.samples;
        sentences = data.sentences;
        settings = data.settings;
        save();
        renderTokens();
        renderSavedSentences();
        modelTrained = false; // Se necesita re-entrenar con los nuevos datos
        updateUiState();
        log('Datos importados correctamente');
      } else {
        alert('El archivo de datos no tiene el formato esperado.');
        log('Error: el archivo de importación no es válido');
      }
    } catch(err) {
      alert('Error al leer el archivo JSON: ' + err.message);
      log('Error de importación: ' + err.message);
    }
  };
  r.readAsText(f);
};
$('#btnReset').onclick = ()=>{ if(confirm('¿Borrar todo?')){ samples={}; sentences=[]; tokens=[]; modelTrained = false; save(); renderTokens(); renderSavedSentences(); log('Reset completo'); updateUiState(); } };

// tokens / oraciones
$('#btnSpace').onclick = ()=>{ tokens.push(''); renderTokens(); };
$('#btnDelete').onclick = ()=>{ tokens.pop(); renderTokens(); };
$('#btnClear').onclick = ()=>{ tokens=[]; renderTokens(); };
$('#btnFinish').onclick = ()=>{ if(tokens.length){ const s = tokens.join(' ').replace(/\s+/g,' ').trim()+'.'; sentences.push({text:s, ts:Date.now()}); tokens=[]; renderTokens(); save(); renderSavedSentences(); log('Oración guardada: '+s); } };
$('#btnComma').onclick = ()=>{ tokens.push(','); renderTokens(); };
$('#btnSaveSentence').onclick = ()=>{ const s = $('#sentence').textContent.trim(); if(!s) return; sentences.push({text:s, ts:Date.now()}); save(); renderSavedSentences(); log('Oración guardada (manual)'); };

// Manejador para la lista de oraciones guardadas
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

// Atajos
document.addEventListener('keydown',(ev)=>{
  if(ev.key===' '){ ev.preventDefault(); $('#btnSpace').click(); }
  if(ev.key==='Backspace'){ $('#btnDelete').click(); }
  if(ev.key==='l'||ev.key==='L'){ $('#btnLearn').click(); }
  if(ev.key==='t'||ev.key==='T'){ $('#btnTranslate').click(); }
  if(ev.key===','){ $('#btnComma').click(); }
  if(ev.key==='.') { $('#btnFinish').click(); }
});

// Carga inicial
load();
renderTokens();
renderSavedSentences();
setStatus('inactivo','dot-idle');
updateUiState();

// ===== Lógica para el cambio de tema =====
const themeToggleBtn = document.getElementById('theme-toggle-btn');
const htmlEl = document.documentElement;

const applyTheme = (theme) => {
  htmlEl.dataset.theme = theme;
  localStorage.setItem('theme', theme);
};

const savedTheme = localStorage.getItem('theme') || 'light';
applyTheme(savedTheme);

themeToggleBtn.addEventListener('click', () => {
  const currentTheme = htmlEl.dataset.theme;
  const newTheme = currentTheme === 'light' ? 'dark' : 'light';
  applyTheme(newTheme);
});
