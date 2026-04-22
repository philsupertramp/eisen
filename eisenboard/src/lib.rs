use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, RwLock};
use std::thread;

fn page_size() -> usize {
    unsafe {
        let ps = libc::sysconf(libc::_SC_PAGESIZE);
        if ps <= 0 {
            4096 // fallback (safe default on most systems)
        } else {
            ps as usize
        }
    }
}
pub fn get_rss_bytes() -> usize {
    let statm = std::fs::read_to_string("/proc/self/statm").unwrap();
    let parts: Vec<&str> = statm.split_whitespace().collect();

    let rss_pages: usize = parts[1].parse().unwrap();
    let page_size = page_size();

    rss_pages * page_size
}


#[derive(Clone)]
pub struct StepRecord {
    pub step: usize,
    pub loss: f32,
    pub grad_norm: f32,
}

#[derive(Clone, Default)]
pub struct TrainStats {
    pub step: usize,
    pub loss: f32,
    pub lr: f32,
    pub tps: f32,
    pub batch_time_ms: f32,
    pub total_tokens: usize,
    pub grad_norm: f32,
    pub grad_clip_coef: f32,
    pub accum_steps: usize,
    pub seq_len: usize,
    pub micro_batch_size: usize,
    pub effective_batch: usize,
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub ffn_dim: usize,
    pub num_layers: usize,
    pub total_params: usize,
    pub history: Vec<StepRecord>,
    pub rss: usize,
    pub gpu_mem: usize,
}

const DASHBOARD_HTML: &str = r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EisenBoard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root {
  --bg:        #07070f;
  --surface:   #0d0d1a;
  --surface2:  #12122a;
  --border:    #1d1d30;
  --amber:     #f59e0b;
  --amber-dim: #78350f;
  --green:     #22c55e;
  --red:       #ef4444;
  --blue:      #60a5fa;
  --text:      #e2e8f0;
  --muted:     #4b5563;
  --mono:      'JetBrains Mono', monospace;
  --display:   'Syne', sans-serif;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--mono);
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
body::before {
  content: '';
  position: fixed;
  inset: 0;
  background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.04) 2px, rgba(0,0,0,0.04) 4px);
  pointer-events: none;
  z-index: 999;
}

/* ── Header ── */
.header {
  display: flex;
  align-items: center;
  gap: 28px;
  padding: 12px 20px;
  border-bottom: 1px solid var(--border);
  background: var(--surface);
  flex-shrink: 0;
}
.logo {
  font-family: var(--display);
  font-size: 17px;
  font-weight: 800;
  color: var(--amber);
  letter-spacing: -0.5px;
  white-space: nowrap;
}
.hstat { display: flex; flex-direction: column; gap: 1px; }
.hstat .lbl { font-size: 8px; color: var(--muted); text-transform: uppercase; letter-spacing: 1.5px; }
.hstat .val { font-size: 14px; font-weight: 600; color: var(--text); }
.spacer { flex: 1; }
.live { display: flex; align-items: center; gap: 7px; font-size: 11px; color: var(--green); }
.dot { width: 6px; height: 6px; border-radius: 50%; background: var(--green); animation: blink 1.6s ease-in-out infinite; }
@keyframes blink { 0%,100%{opacity:1;} 50%{opacity:0.25;} }

/* ── Layout ── */
.body {
  display: grid;
  grid-template-columns: 1fr 290px;
  grid-template-rows: 1fr auto;
  gap: 10px;
  padding: 10px;
  flex: 1;
  min-height: 0;
}

/* ── Panels ── */
.panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 14px;
  overflow: hidden;
}
.ptitle {
  font-size: 8px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 2px;
  margin-bottom: 10px;
}

/* ── Chart panel ── */
.chart-panel {
  grid-row: 1;
  display: flex;
  flex-direction: column;
}
.chart-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
}
.chart-header .ptitle { margin-bottom: 0; }
.loss-tag {
  font-size: 9px;
  padding: 2px 7px;
  border-radius: 3px;
  background: var(--amber-dim);
  color: var(--amber);
}
.chart-wrap { flex: 1; position: relative; min-height: 0; }

/* ── Side panel ── */
.side {
  grid-row: 1;
  display: flex;
  flex-direction: column;
  gap: 8px;
  overflow-y: auto;
  scrollbar-width: none;
}

/* ── Stat rows ── */
.row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 3px 0;
  border-bottom: 1px solid var(--border);
  font-size: 11.5px;
}
.row:last-child { border-bottom: none; }
.row .k { color: var(--muted); }
.row .v { font-weight: 500; }
.row .v.a { color: var(--amber); }
.row .v.g { color: var(--green); }
.row .v.r { color: var(--red); }

/* ── Mini grad chart ── */
.grad-wrap { height: 52px; margin-top: 8px; }

/* ── Bottom insight strip ── */
.strip {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 8px;
}
.card {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 9px 12px;
}
.card .lbl { font-size: 8px; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 3px; }
.card .val { font-size: 15px; font-weight: 600; color: var(--amber); }
.card .sub { font-size: 9px; color: var(--muted); margin-top: 2px; }
.card .val.g { color: var(--green); }
.card .val.r { color: var(--red); }
</style>
</head>
<body>

<div class="header">
  <span class="logo">⬡ EisenBoard</span>
  <div class="hstat"><span class="lbl">Step</span><span class="val" id="h-step">—</span></div>
  <div class="hstat"><span class="lbl">Loss</span><span class="val" id="h-loss">—</span></div>
  <div class="hstat"><span class="lbl">LR</span><span class="val" id="h-lr">—</span></div>
  <div class="hstat"><span class="lbl">TPS</span><span class="val" id="h-tps">—</span></div>
  <div class="hstat"><span class="lbl">Steps/s</span><span class="val" id="h-sps">—</span></div>
  <div class="spacer"></div>
  <div class="live"><div class="dot"></div>LIVE</div>
</div>

<div class="body">

  <!-- LOSS CHART -->
  <div class="panel chart-panel">
    <div class="chart-header">
      <span class="ptitle">Training Loss</span>
      <span class="loss-tag" id="loss-tag">—</span>
    </div>
    <div class="chart-wrap">
      <canvas id="loss-chart"></canvas>
    </div>
  </div>

  <!-- SIDE -->
  <div class="side">

    <div class="panel">
      <div class="ptitle">Runtime</div>
      <div class="row"><span class="k">Batch time</span><span class="v" id="s-bat">—</span></div>
      <div class="row"><span class="k">Tokens / step</span><span class="v a" id="s-tpstep">—</span></div>
      <div class="row"><span class="k">Total tokens</span><span class="v" id="s-tok">—</span></div>
      <div class="row"><span class="k">Effective batch</span><span class="v" id="s-eff">—</span></div>
      <div class="row"><span class="k">Micro × Accum</span><span class="v" id="s-mic">—</span></div>
      <div class="row"><span class="k">Seq len</span><span class="v" id="s-seq">—</span></div>
    </div>

    <div class="panel">
      <div class="ptitle">Gradients</div>
      <div class="row"><span class="k">Norm</span><span class="v" id="s-gn">—</span></div>
      <div class="row"><span class="k">Clip coef</span><span class="v" id="s-gc">—</span></div>
      <div class="row"><span class="k">Norm trend</span><span class="v" id="s-gt">—</span></div>
      <div class="grad-wrap"><canvas id="grad-chart"></canvas></div>
    </div>

    <div class="panel">
      <div class="ptitle">Architecture</div>
      <div class="row"><span class="k">Parameters</span><span class="v a" id="s-par">—</span></div>
      <div class="row"><span class="k">Size fp16</span><span class="v" id="s-sz">—</span></div>
      <div class="row"><span class="k">Size fp32</span><span class="v" id="s-sz32">—</span></div>
      <div class="row"><span class="k">Layers / Heads</span><span class="v" id="s-lh">—</span></div>
      <div class="row"><span class="k">Hidden / FFN</span><span class="v" id="s-hf">—</span></div>
      <div class="row"><span class="k">Vocab</span><span class="v" id="s-voc">—</span></div>
    </div>

    <div class="panel">
      <div class="ptitle">Memory</div>
      <div class="row"><span class="k">RAM (RSS)</span><span class="v" id="s-rss">—</span></div>
      <div class="row"><span class="k">GPU VRAM</span><span class="v" id="s-gpu">—</span></div>
    </div>

  </div>

  <!-- COMPUTED INSIGHT STRIP -->
  <div class="strip">
    <div class="card">
      <div class="lbl">Loss EMA</div>
      <div class="val" id="c-ema">—</div>
      <div class="sub">α = 0.05 smoothed</div>
    </div>
    <div class="card">
      <div class="lbl">Loss Δ 100 steps</div>
      <div class="val" id="c-ld">—</div>
      <div class="sub">recent trend</div>
    </div>
    <div class="card">
      <div class="lbl">Loss / B tokens</div>
      <div class="val" id="c-lbt">—</div>
      <div class="sub">Chinchilla proxy</div>
    </div>
    <div class="card">
      <div class="lbl">Steps / sec</div>
      <div class="val" id="c-sps">—</div>
      <div class="sub">wall-clock throughput</div>
    </div>
    <div class="card">
      <div class="lbl">Tokens / step</div>
      <div class="val" id="c-tks">—</div>
      <div class="sub">eff_batch × seq_len</div>
    </div>
    <div class="card">
      <div class="lbl">Grad / Loss</div>
      <div class="val" id="c-gl">—</div>
      <div class="sub">stability proxy</div>
    </div>
  </div>

</div>

<script>
Chart.defaults.color = '#4b5563';
Chart.defaults.borderColor = '#1d1d30';

// ── Loss chart ──
const lossChart = new Chart(document.getElementById('loss-chart').getContext('2d'), {
  type: 'line',
  data: {
    labels: [],
    datasets: [
      {
        label: 'Loss',
        data: [],
        borderColor: '#f59e0b',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.3,
        fill: true,
        backgroundColor: 'rgba(245,158,11,0.06)',
      },
      {
        label: 'EMA',
        data: [],
        borderColor: '#60a5fa',
        borderWidth: 1.5,
        borderDash: [5, 4],
        pointRadius: 0,
        tension: 0.3,
        fill: false,
      }
    ]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 0 },
    plugins: {
      legend: { labels: { font: { family: 'JetBrains Mono', size: 11 }, boxWidth: 14, padding: 16 } },
      tooltip: {
        mode: 'index', intersect: false,
        backgroundColor: '#0d0d1a', borderColor: '#1d1d30', borderWidth: 1,
        titleFont: { family: 'JetBrains Mono', size: 11 },
        bodyFont: { family: 'JetBrains Mono', size: 11 },
        callbacks: { label: ctx => ' ' + ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(4) }
      }
    },
    scales: {
      x: { ticks: { maxTicksLimit: 8, font: { family: 'JetBrains Mono', size: 10 } }, grid: { color: '#0f0f1e' } },
      y: { ticks: { font: { family: 'JetBrains Mono', size: 10 } }, grid: { color: '#0f0f1e' } }
    }
  }
});

// ── Grad norm sparkline ──
const gradChart = new Chart(document.getElementById('grad-chart').getContext('2d'), {
  type: 'line',
  data: {
    labels: [],
    datasets: [{
      data: [],
      borderColor: '#22c55e',
      borderWidth: 1.5,
      pointRadius: 0,
      tension: 0.3,
      fill: true,
      backgroundColor: 'rgba(34,197,94,0.07)',
    }]
  },
  options: {
    responsive: true, maintainAspectRatio: false, animation: { duration: 0 },
    plugins: { legend: { display: false }, tooltip: { enabled: false } },
    scales: { x: { display: false }, y: { display: false } }
  }
});

// ── Helpers ──
function computeEMA(arr, a) {
  if (!arr.length) return [];
  const out = [arr[0]];
  for (let i = 1; i < arr.length; i++) out.push(a * arr[i] + (1 - a) * out[i-1]);
  return out;
}

function fmt(n, d=2) {
  if (n == null) return '—';
  if (n >= 1e9) return (n/1e9).toFixed(d) + 'B';
  if (n >= 1e6) return (n/1e6).toFixed(d) + 'M';
  if (n >= 1e3) return (n/1e3).toFixed(d) + 'K';
  return typeof n === 'number' ? n.toFixed(d) : n;
}

function set(id, text, cls) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = text;
  if (cls !== undefined) el.className = 'val ' + cls;
}

function setRow(id, text, cls) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = text;
  if (cls !== undefined) el.className = 'v ' + cls;
}

// ── Poll ──
setInterval(async () => {
  try {
    const d = await fetch('/api/stats').then(r => r.json());
    const hist = d.history || [];
    const losses = hist.map(h => h.loss);
    const gnorms = hist.map(h => h.grad_norm);

    // Header
    document.getElementById('h-step').textContent = d.step.toLocaleString();
    document.getElementById('h-loss').textContent = d.loss.toFixed(4);
    document.getElementById('h-lr').textContent = d.lr.toExponential(2);
    document.getElementById('h-tps').textContent = Math.round(d.tps).toLocaleString();
    document.getElementById('h-sps').textContent = d.batch_time_ms > 0 ? (1000/d.batch_time_ms).toFixed(2) : '—';

    // Runtime
    const tokPerStep = d.effective_batch * d.seq_len;
    setRow('s-bat', Math.round(d.batch_time_ms) + ' ms');
    setRow('s-tpstep', fmt(tokPerStep, 1));
    setRow('s-tok', fmt(d.total_tokens, 2));
    setRow('s-eff', d.effective_batch.toLocaleString());
    setRow('s-mic', d.micro_batch_size + ' × ' + d.accum_steps);
    setRow('s-seq', d.seq_len.toLocaleString());

    // Gradients
    setRow('s-gn', d.grad_norm.toFixed(4));
    setRow('s-gc', d.grad_clip_coef.toFixed(4));

    if (gnorms.length >= 20) {
      const recent = gnorms.slice(-10).reduce((a,b)=>a+b,0)/10;
      const older  = gnorms.slice(-20,-10).reduce((a,b)=>a+b,0)/10;
      if (recent < older * 0.95)      setRow('s-gt', '↓ decreasing', 'g');
      else if (recent > older * 1.05) setRow('s-gt', '↑ increasing', 'r');
      else                             setRow('s-gt', '→ stable', '');
    }

    // Architecture
    setRow('s-par', fmt(d.total_params, 2));
    setRow('s-sz', (d.total_params * 2 / 1e9).toFixed(2) + ' GB');
    setRow('s-sz32', (d.total_params * 4 / 1e9).toFixed(2) + ' GB');
    setRow('s-lh', d.num_layers + ' / ' + d.num_heads);
    setRow('s-hf', d.hidden_dim + ' / ' + d.ffn_dim);
    setRow('s-voc', fmt(d.vocab_size, 1));

    // Memory
    if (d.rss)     setRow('s-rss', (d.rss/1e9).toFixed(2) + ' GB');
    if (d.gpu_mem) setRow('s-gpu', (d.gpu_mem/1e9).toFixed(2) + ' GB');

    // ── Computed insights ──
    const ema = computeEMA(losses, 0.05);
    set('c-ema', ema.length ? ema[ema.length-1].toFixed(4) : '—');

    if (hist.length >= 2) {
      const lb = Math.min(100, hist.length - 1);
      const delta = hist[hist.length-1].loss - hist[hist.length-1-lb].loss;
      const sign = delta < 0 ? '' : '+';
      set('c-ld', sign + delta.toFixed(4), delta < 0 ? 'g' : 'r');
    }

    set('c-lbt', d.total_tokens > 0 ? (d.loss / (d.total_tokens / 1e9)).toFixed(3) : '—');
    set('c-sps', d.batch_time_ms > 0 ? (1000/d.batch_time_ms).toFixed(2) : '—');
    set('c-tks', fmt(tokPerStep, 1));
    set('c-gl',  d.loss > 0 ? (d.grad_norm / d.loss).toFixed(4) : '—');

    // Loss % drop tag
    if (hist.length >= 2) {
      const pct = ((hist[0].loss - hist[hist.length-1].loss) / hist[0].loss * 100).toFixed(1);
      document.getElementById('loss-tag').textContent = '↓ ' + pct + '% from start';
    }

    // ── Charts ──
    const MAX = 600;
    const stride = Math.max(1, Math.floor(hist.length / MAX));
    const sh = hist.filter((_,i) => i % stride === 0);
    const se = computeEMA(losses, 0.05).filter((_,i) => i % stride === 0);

    lossChart.data.labels = sh.map(h => h.step);
    lossChart.data.datasets[0].data = sh.map(h => h.loss);
    lossChart.data.datasets[1].data = se;
    lossChart.update('none');

    const gs = Math.max(1, Math.floor(hist.length / 120));
    const sg = hist.filter((_,i) => i % gs === 0);
    gradChart.data.labels = sg.map(h => h.step);
    gradChart.data.datasets[0].data = sg.map(h => h.grad_norm);
    gradChart.update('none');

  } catch(e) { console.error(e); }
}, 1000);
</script>
</body>
</html>
"##;

pub fn spawn_eisenboard(stats: Arc<RwLock<TrainStats>>, bind_addr: &str) {
    let bind_addr = bind_addr.to_string();
    thread::spawn(move || {
        let listener = TcpListener::bind(&bind_addr)
            .unwrap_or_else(|_| panic!("Failed to bind EisenBoard to {}", bind_addr));
        println!("\n🌐 EisenBoard Live! Open http://{} in your browser\n", bind_addr);
        for stream in listener.incoming() {
            if let Ok(mut stream) = stream {
                let _ = stream.set_read_timeout(Some(std::time::Duration::from_millis(500)));
                let mut buffer = [0; 1024];
                if stream.read(&mut buffer).unwrap_or(0) > 0 {
                    let request = String::from_utf8_lossy(&buffer[..]);
                    if request.starts_with("GET /api/stats") {
                        let s = stats.read().unwrap();
                        let history_json: Vec<String> = s
                            .history
                            .iter()
                            .map(|r| {
                                format!(
                                    r#"{{"step":{},"loss":{:.6},"grad_norm":{:.6}}}"#,
                                    r.step, r.loss, r.grad_norm
                                )
                            })
                            .collect();
                        let json = format!(
                            r#"{{"step":{},"loss":{:.6},"lr":{:.8},"tps":{:.2},"batch_time_ms":{:.2},"total_tokens":{},"grad_norm":{:.6},"grad_clip_coef":{:.6},"accum_steps":{},"seq_len":{},"micro_batch_size":{},"effective_batch":{},"vocab_size":{},"hidden_dim":{},"num_heads":{},"ffn_dim":{},"num_layers":{},"total_params":{},"rss":{},"gpu_mem":{},"history":[{}]}}"#,
                            s.step,
                            s.loss,
                            s.lr,
                            s.tps,
                            s.batch_time_ms,
                            s.total_tokens,
                            s.grad_norm,
                            s.grad_clip_coef,
                            s.accum_steps,
                            s.seq_len,
                            s.micro_batch_size,
                            s.effective_batch,
                            s.vocab_size,
                            s.hidden_dim,
                            s.num_heads,
                            s.ffn_dim,
                            s.num_layers,
                            s.total_params,
                            s.rss,
                            s.gpu_mem,
                            history_json.join(",")
                        );
                        let response = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}",
                            json
                        );
                        let _ = stream.write_all(response.as_bytes());
                    } else if request.starts_with("GET / ") {
                        let response = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n{}",
                            DASHBOARD_HTML
                        );
                        let _ = stream.write_all(response.as_bytes());
                    }
                }
            }
        }
    });
}
