use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, RwLock};
use std::thread;

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
}

const DASHBOARD_HTML: &str = r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>EisenBoard | Live Training</title>
    <style>
        body { background-color: #0d1117; color: #c9d1d9; font-family: monospace; padding: 2rem; margin: 0; }
        h1 { color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 10px; }
        .stats-grid { display: flex; gap: 15px; margin-bottom: 20px; flex-wrap: wrap; max-width: 1200px; }
        .stat-box { background: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 6px; flex: 1 1 170px; }
        .stat-label { font-size: 12px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
        .stat-value { font-size: 24px; color: #7ee787; font-weight: bold; margin-top: 5px; }
        .stat-value.lr { color: #f78166; }
        .stat-value.warn { color: #ffa657; }
        canvas { background: #161b22; border: 1px solid #30363d; border-radius: 6px; width: 100%; max-width: 1200px; height: 400px; }
    </style>
</head>
<body>
    <h1>EisenBoard 🚀</h1>
    <div class="stats-grid">
        <div class="stat-box"><div class="stat-label">STEP</div><div id="step" class="stat-value">0</div></div>
        <div class="stat-box"><div class="stat-label">LOSS</div><div id="loss" class="stat-value">0.0000</div></div>
        <div class="stat-box"><div class="stat-label">LR</div><div id="lr" class="stat-value lr">0.0000</div></div>
        <div class="stat-box"><div class="stat-label">TOKENS / SEC</div><div id="tps" class="stat-value">0</div></div>
        <div class="stat-box"><div class="stat-label">BATCH TIME</div><div id="batch_time" class="stat-value">0 ms</div></div>
        <div class="stat-box"><div class="stat-label">TOTAL TOKENS</div><div id="total_tokens" class="stat-value">0</div></div>
        <div class="stat-box"><div class="stat-label">GRAD NORM</div><div id="grad_norm" class="stat-value warn">0.00</div></div>
        <div class="stat-box"><div class="stat-label">CLIP COEF</div><div id="grad_clip_coef" class="stat-value warn">1.00</div></div>
        <div class="stat-box"><div class="stat-label">MICRO / ACCUM</div><div id="batch_cfg" class="stat-value">0 / 0</div></div>
        <div class="stat-box"><div class="stat-label">SEQ LEN</div><div id="seq_len" class="stat-value">0</div></div>
        <div class="stat-box"><div class="stat-label">MODEL PARAMS</div><div id="total_params" class="stat-value">0</div></div>
        <div class="stat-box"><div class="stat-label">LAYERS / HEADS</div><div id="arch_lh" class="stat-value">0 / 0</div></div>
        <div class="stat-box"><div class="stat-label">HIDDEN / FFN</div><div id="arch_hf" class="stat-value">0 / 0</div></div>
        <div class="stat-box"><div class="stat-label">VOCAB</div><div id="vocab_size" class="stat-value">0</div></div>
    </div>
    <canvas id="lossChart" width="1200" height="400"></canvas>
    <script>
        const ctx = document.getElementById('lossChart').getContext('2d');
        function drawChart(history) {
            ctx.clearRect(0, 0, 1200, 400);
            if (!history || history.length < 2) return;
            ctx.strokeStyle = '#30363d'; ctx.lineWidth = 1;
            for(let i=0; i<=10; i++) { ctx.beginPath(); ctx.moveTo(0, i*40); ctx.lineTo(1200, i*40); ctx.stroke(); }
            let minLoss = Math.min(...history.map(d => d.loss)) * 0.98;
            let maxLoss = Math.max(...history.map(d => d.loss)) * 1.02;
            let range = maxLoss - minLoss || 1;
            ctx.strokeStyle = '#ff7b72'; ctx.lineWidth = 2; ctx.beginPath();
            history.forEach((point, i) => {
                let x = (i / Math.max(history.length - 1, 1)) * 1200;
                let y = 400 - (((point.loss - minLoss) / range) * 400);
                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            });
            ctx.stroke();
        }
        setInterval(async () => {
            try {
                const res = await fetch('/api/stats');
                const data = await res.json();
                document.getElementById('step').innerText = data.step.toLocaleString();
                document.getElementById('loss').innerText = data.loss.toFixed(4);
                document.getElementById('lr').innerText = data.lr.toExponential(2);
                document.getElementById('tps').innerText = Math.round(data.tps).toLocaleString();
                document.getElementById('batch_time').innerText = Math.round(data.batch_time_ms) + " ms";
                document.getElementById('grad_norm').innerText = data.grad_norm.toFixed(4);
                document.getElementById('grad_clip_coef').innerText = data.grad_clip_coef.toFixed(4);
                document.getElementById('batch_cfg').innerText = `${data.micro_batch_size} / ${data.accum_steps}`;
                document.getElementById('seq_len').innerText = data.seq_len.toLocaleString();
                document.getElementById('arch_lh').innerText = `${data.num_layers} / ${data.num_heads}`;
                document.getElementById('arch_hf').innerText = `${data.hidden_dim} / ${data.ffn_dim}`;
                document.getElementById('vocab_size').innerText = data.vocab_size.toLocaleString();
                let tk = data.total_tokens;
                let tkStr = tk > 1000000000 ? (tk/1000000000).toFixed(2) + "B" : tk > 1000000 ? (tk/1000000).toFixed(2) + "M" : tk.toLocaleString();
                document.getElementById('total_tokens').innerText = tkStr;
                let p = data.total_params;
                let pStr = p > 1000000000 ? (p/1000000000).toFixed(2) + "B" : p > 1000000 ? (p/1000000).toFixed(2) + "M" : p.toLocaleString();
                document.getElementById('total_params').innerText = pStr;
                drawChart(data.history);
            } catch (e) {}
        }, 1000);
    </script>
</body>
</html>
</html>
"#;

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
                            .map(|r| format!(r#"{{"step":{},"loss":{:.6}}}"#, r.step, r.loss))
                            .collect();
                        let json = format!(
                            r#"{{"step":{},"loss":{:.6},"lr":{:.8},"tps":{:.2},"batch_time_ms":{:.2},"total_tokens":{},"grad_norm":{:.6},"grad_clip_coef":{:.6},"accum_steps":{},"seq_len":{},"micro_batch_size":{},"effective_batch":{},"vocab_size":{},"hidden_dim":{},"num_heads":{},"ffn_dim":{},"num_layers":{},"total_params":{},"history":[{}]}}"#,
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
