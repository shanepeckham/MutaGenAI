// MutaGenAI Evolution Dashboard — webview script
// Loaded as an external file via <script src="...">

// Immediately change banner to prove JS runs
document.getElementById('jsBanner').textContent = '✓ JS LOADED at ' + new Date().toISOString();
document.getElementById('jsBanner').style.background = '#4ec9b0';

window.onerror = function(msg, url, line, col, err) {
  var b = document.getElementById('jsBanner');
  if (b) {
    b.textContent = 'JS ERROR: ' + msg + ' at line ' + line;
    b.style.background = '#f14c4c';
  }
};

// ── State ─────────────────────────────────────────────────────────────
var vscode;
try {
  vscode = acquireVsCodeApi();
  document.getElementById('jsBanner').textContent += ' | vscode API OK';
} catch(ex) {
  document.getElementById('jsBanner').textContent = 'FATAL: acquireVsCodeApi failed: ' + ex;
  document.getElementById('jsBanner').style.background = '#f14c4c';
}
var state = {
  generations: [],
  candidates: [],
  lineageNodes: [],
  bestScore: 0,
  bestHash: '',
  bestPrompt: '',
  totalGen: 0,
  currentGen: 0,
  startTime: Date.now(),
  status: 'idle',
};

// ── Utility ───────────────────────────────────────────────────────────
function ts() {
  var d = new Date();
  return d.toLocaleTimeString('en-GB', { hour12: false });
}

function addLog(msg, level) {
  var box = document.getElementById('logBox');
  var div = document.createElement('div');
  div.className = 'log-line ' + (level || '');
  div.innerHTML = '<span class="ts">' + ts() + '</span>' + escHtml(msg);
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
  while (box.children.length > 200) box.removeChild(box.firstChild);
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function setStatus(s) {
  state.status = s;
  var badge = document.getElementById('statusBadge');
  badge.className = 'badge ' + s;
  badge.textContent = s === 'running' ? 'Running' :
                      s === 'done' ? 'Complete' :
                      s === 'stopped' ? 'Stopped' :
                      s === 'error' ? 'Error' : 'Idle';
  document.getElementById('stopBtn').style.display =
    s === 'running' ? 'inline-block' : 'none';
}

function doStop() {
  vscode.postMessage({ command: 'stop' });
  addLog('Stop requested…', 'warn');
}

// ── Chart drawing (pure canvas — no deps) ─────────────────────────────
function drawChart() {
  var canvas = document.getElementById('scoreChart');
  var ctx = canvas.getContext('2d');
  var dpr = window.devicePixelRatio || 1;
  var rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  var W = rect.width, H = rect.height;
  var pad = { top: 14, right: 14, bottom: 30, left: 44 };
  var cw = W - pad.left - pad.right;
  var ch = H - pad.top - pad.bottom;

  ctx.clearRect(0, 0, W, H);

  if (state.generations.length === 0) {
    ctx.fillStyle = getComputedStyle(document.body).getPropertyValue('--fg') || '#ccc';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Waiting for data…', W/2, H/2);
    return;
  }

  var maxGen = state.totalGen || state.generations.length;
  var maxScore = 100;

  // Grid
  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineWidth = 1;
  for (var i = 0; i <= 4; i++) {
    var y = pad.top + ch - (i/4) * ch;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + cw, y); ctx.stroke();
  }

  // Axes
  var fg = getComputedStyle(document.body).getPropertyValue('--fg') || '#ccc';
  ctx.strokeStyle = fg;
  ctx.globalAlpha = 0.3;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, pad.top + ch);
  ctx.lineTo(pad.left + cw, pad.top + ch);
  ctx.stroke();
  ctx.globalAlpha = 1;

  // Labels
  ctx.fillStyle = fg;
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'right';
  for (var i = 0; i <= 4; i++) {
    var y = pad.top + ch - (i/4) * ch;
    ctx.fillText((i * 25) + '%', pad.left - 6, y + 3);
  }
  ctx.textAlign = 'center';
  var genStep = Math.max(1, Math.floor(maxGen / 6));
  for (var g = 0; g <= maxGen; g += genStep) {
    var x = pad.left + (g / maxGen) * cw;
    ctx.fillText(g.toString(), x, pad.top + ch + 16);
  }
  ctx.fillText('Generation', pad.left + cw / 2, H - 4);

  // Best score line
  var accent = getComputedStyle(document.body).getPropertyValue('--accent') || '#4fc1ff';
  ctx.strokeStyle = accent;
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (var i = 0; i < state.generations.length; i++) {
    var g = state.generations[i];
    var x = pad.left + (g.gen / maxGen) * cw;
    var y = pad.top + ch - (g.bestScore / maxScore) * ch;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Dots
  ctx.fillStyle = accent;
  for (var i = 0; i < state.generations.length; i++) {
    var g = state.generations[i];
    var x = pad.left + (g.gen / maxGen) * cw;
    var y = pad.top + ch - (g.bestScore / maxScore) * ch;
    ctx.beginPath();
    ctx.arc(x, y, g.improved ? 5 : 3, 0, Math.PI * 2);
    ctx.fill();
  }

  // Gen-best score line (fainter)
  ctx.strokeStyle = 'rgba(78,201,176,0.4)';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  for (var i = 0; i < state.generations.length; i++) {
    var g = state.generations[i];
    var x = pad.left + (g.gen / maxGen) * cw;
    var y = pad.top + ch - (g.genBestScore / maxScore) * ch;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.setLineDash([]);
}

// ── Lineage tree drawing (pure SVG) ───────────────────────────────────
var islandColors = ['#4fc1ff','#c586c0','#4ec9b0','#ce9178','#dcdcaa',
                    '#569cd6','#d7ba7d','#b5cea8','#f44747','#9cdcfe'];

function drawLineage() {
  var container = document.getElementById('lineageTree');
  var nodes = state.lineageNodes;
  if (nodes.length === 0) {
    container.innerHTML = '<div style="opacity:0.4;text-align:center;margin-top:40px;">No candidates yet</div>';
    return;
  }

  // Group by generation
  var byGen = {};
  var maxGen = 0;
  for (var ni = 0; ni < nodes.length; ni++) {
    var n = nodes[ni];
    if (!byGen[n.generation]) byGen[n.generation] = [];
    byGen[n.generation].push(n);
    if (n.generation > maxGen) maxGen = n.generation;
  }

  var nodeR = 8;
  var colW = 60;
  var rowH = 28;
  var padX = 30;
  var padY = 24;

  // Assign positions
  var positions = {};
  var maxY = 0;
  for (var g = 0; g <= maxGen; g++) {
    var gNodes = byGen[g] || [];
    gNodes.sort(function(a, b) { return b.score - a.score; });
    var displayed = gNodes.slice(0, 16);
    for (var i = 0; i < displayed.length; i++) {
      var x = padX + g * colW;
      var y = padY + i * rowH;
      positions[displayed[i].hash] = { x: x, y: y };
      if (y > maxY) maxY = y;
    }
  }

  var svgW = padX + (maxGen + 1) * colW + 20;
  var svgH = maxY + padY + 20;

  var svg = '<svg xmlns="http://www.w3.org/2000/svg" width="' + svgW + '" height="' + svgH + '">';

  // Links
  for (var ni = 0; ni < nodes.length; ni++) {
    var n = nodes[ni];
    var pos = positions[n.hash];
    if (!pos) continue;
    var phs = n.parentHashes || [];
    for (var pi = 0; pi < phs.length; pi++) {
      var ppos = positions[phs[pi]];
      if (!ppos) continue;
      svg += '<line class="link" x1="' + ppos.x + '" y1="' + ppos.y +
             '" x2="' + pos.x + '" y2="' + pos.y + '"/>';
    }
  }

  // Nodes
  for (var ni = 0; ni < nodes.length; ni++) {
    var n = nodes[ni];
    var pos = positions[n.hash];
    if (!pos) continue;
    var col = islandColors[n.island % islandColors.length];
    var isBest = n.hash === state.bestHash;
    var r = isBest ? nodeR + 3 : nodeR;
    var strokeW = isBest ? 3 : 1.5;
    svg += '<g class="node" data-hash="' + n.hash + '"' +
           ' onmouseenter="showTip(event,\'' + n.hash + '\')"' +
           ' onmouseleave="hideTip()">';
    svg += '<circle cx="' + pos.x + '" cy="' + pos.y + '" r="' + r +
           '" fill="' + col + '" stroke="' + (isBest ? '#fff' : col) +
           '" stroke-width="' + strokeW + '" opacity="' + (0.4 + 0.6 * n.score / 100) + '"/>';
    svg += '<text x="' + pos.x + '" y="' + (pos.y - r - 3) +
           '" text-anchor="middle" font-size="8" opacity="0.6">' +
           n.score.toFixed(0) + '</text>';
    svg += '</g>';
  }

  // Generation labels
  for (var g = 0; g <= maxGen; g++) {
    var x = padX + g * colW;
    svg += '<text x="' + x + '" y="12" text-anchor="middle" font-size="9" ' +
           'fill="' + (getComputedStyle(document.body).getPropertyValue('--fg') || '#ccc') +
           '" opacity="0.5">G' + g + '</text>';
  }

  svg += '</svg>';
  container.innerHTML = svg;
}

// Tooltip for lineage nodes
function showTip(ev, hash) {
  var node = null;
  for (var i = 0; i < state.lineageNodes.length; i++) {
    if (state.lineageNodes[i].hash === hash) { node = state.lineageNodes[i]; break; }
  }
  if (!node) return;
  var tip = document.getElementById('tooltip');
  tip.innerHTML =
    '<div class="tt-label">Hash</div><div class="tt-value">' + node.hash + '</div>' +
    '<div class="tt-label">Score</div><div class="tt-value">' + node.score.toFixed(1) + '%</div>' +
    '<div class="tt-label">Operation</div><div class="tt-value">' + escHtml(node.operation) + '</div>' +
    '<div class="tt-label">Island</div><div class="tt-value">' + node.island + '</div>' +
    '<div class="tt-label">Temp / Top-p</div><div class="tt-value">' +
      node.temperature.toFixed(3) + ' / ' + node.topP.toFixed(3) + '</div>' +
    '<div class="tt-label" style="margin-top:4px">Prompt (preview)</div>' +
    '<div style="font-size:10px;opacity:0.8;max-height:80px;overflow:hidden;word-break:break-word;">' +
      escHtml((node.template || '').substring(0, 200)) + '</div>';
  tip.style.display = 'block';
  tip.style.left = Math.min(ev.pageX + 10, document.body.clientWidth - 340) + 'px';
  tip.style.top = (ev.pageY + 10) + 'px';
}
function hideTip() {
  document.getElementById('tooltip').style.display = 'none';
}

// ── Message handler ───────────────────────────────────────────────────
var msgCount = 0;
window.addEventListener('message', function(ev) {
  try {
  var e = ev.data;
  msgCount++;
  if (!e || !e.type) return;

  if (e.type === '_readyAck') {
    _readyAcked = true;
    addLog('Connected to extension \u2014 receiving events', '');
    return;
  }

  addLog('[event] ' + e.type, '');

  switch (e.type) {
    case 'started':
      state.startTime = Date.now();
      state.generations = [];
      state.candidates = [];
      state.lineageNodes = [];
      state.bestScore = 0;
      state.bestHash = '';
      state.bestPrompt = '';
      state.totalGen = (e.config && e.config.iterations) || 0;
      state.currentGen = 0;
      setStatus('running');
      addLog('Evolution started \u2014 ' + state.totalGen + ' generations', '');
      drawChart();
      drawLineage();
      break;

    case 'status':
      addLog(e.message, '');
      break;

    case 'log':
      addLog(e.message, e.level || '');
      break;

    case 'seed':
      state.lineageNodes.push({
        hash: e.hash, parentHashes: [], operation: 'seed',
        generation: 0, island: e.island, score: e.score,
        temperature: 0, topP: 0, template: e.template,
      });
      addLog('Seed ' + (e.index + 1) + '/' + e.total +
             ' \u2192 island ' + e.island + '  score=' + e.score.toFixed(1) + '%', '');
      document.getElementById('candidateCount').textContent = String(state.lineageNodes.length);
      break;

    case 'seedComplete':
      state.bestScore = e.bestScore;
      state.bestHash = e.bestHash;
      document.getElementById('bestScore').textContent = e.bestScore.toFixed(1) + '%';
      addLog('Seed evaluation complete \u2014 best=' + e.bestScore.toFixed(1) + '%', '');
      drawLineage();
      break;

    case 'generation':
      state.currentGen = e.generation;
      state.bestScore = e.bestScore;
      state.bestHash = e.bestHash;
      state.generations.push({
        gen: e.generation,
        bestScore: e.bestScore,
        genBestScore: e.genBestScore,
        improved: e.improved,
      });

      var candidates = e.candidates || [];
      for (var ci = 0; ci < candidates.length; ci++) {
        var c = candidates[ci];
        state.lineageNodes.push({
          hash: c.hash,
          parentHashes: c.parentHashes || [],
          operation: c.operation,
          generation: e.generation,
          island: c.island,
          score: c.score,
          temperature: c.temperature,
          topP: c.topP,
          template: c.template,
        });
      }

      var scoreEl = document.getElementById('bestScore');
      scoreEl.textContent = e.bestScore.toFixed(1) + '%';
      scoreEl.className = 'value' + (e.improved ? ' improved' : '');
      document.getElementById('genNum').textContent = e.generation + '/' + e.totalGenerations;
      document.getElementById('candidateCount').textContent = String(e.candidateCount);
      document.getElementById('genLabel').textContent =
        (e.improved ? '\u25B2 New best! ' : '') +
        'Gen ' + e.generation + '/' + e.totalGenerations;

      var pct = (e.generation / e.totalGenerations) * 100;
      document.getElementById('progressBar').style.width = pct + '%';

      var secs = ((Date.now() - state.startTime) / 1000).toFixed(0);
      document.getElementById('elapsed').textContent = secs + 's';

      addLog('Gen ' + e.generation + '/' + e.totalGenerations +
             '  best=' + e.bestScore.toFixed(1) + '%' +
             (e.improved ? ' \u25B2' : '') +
             (e.migrated ? ' [migration]' : ''), '');

      drawChart();
      drawLineage();
      break;

    case 'stopped':
      setStatus('stopped');
      document.getElementById('progressBar').style.width = '100%';
      document.getElementById('progressBar').style.background = 'var(--orange)';
      addLog('Stopped: ' + (e.message || e.reason), 'warn');
      break;

    case 'done':
      setStatus('done');
      document.getElementById('progressBar').style.width = '100%';
      document.getElementById('progressBar').style.background = 'var(--green)';
      state.bestPrompt = e.bestPrompt;
      state.bestScore = e.bestScore;
      document.getElementById('bestScore').textContent = e.bestScore.toFixed(1) + '%';
      document.getElementById('bestPrompt').textContent = e.bestPrompt;
      document.getElementById('elapsed').textContent = e.wallTime.toFixed(1) + 's';

      if (e.lineage && e.lineage.length) {
        state.lineageNodes = e.lineage;
        drawLineage();
      }

      addLog('Evolution complete \u2014 best=' + e.bestScore.toFixed(1) +
             '% in ' + e.wallTime.toFixed(1) + 's  (' +
             e.totalCandidates + ' candidates)', '');
      break;

    case 'error':
      setStatus('error');
      addLog('ERROR: ' + e.message, 'error');
      break;
  }
  } catch (err) {
    addLog('JS error handling event: ' + err, 'error');
  }
});

// Initial render
drawChart();

// Signal to the extension that the webview is ready
var _readyAcked = false;
function _sendReady() {
  if (!_readyAcked) {
    vscode.postMessage({ command: 'ready' });
    setTimeout(_sendReady, 100);
  }
}
_sendReady();
addLog('Dashboard loaded \u2014 waiting for evolution events\u2026', '');
