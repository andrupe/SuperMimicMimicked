"""Auto-refreshing live dashboard for exp4 vs exp6_v2 vs exp8_v2."""

import http.server
import io
import os
import socketserver
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np

if os.path.exists("mlflow.db"):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

CLIENT = MlflowClient()
EXP4_ID = "0ec8874d6eb746cd9236201dfa09937b"


def get_latest_run(name: str) -> str | None:
    runs = mlflow.search_runs(
        search_all_experiments=True,
        filter_string=f"tags.mlflow.runName = '{name}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if len(runs) == 0:
        return None
    return str(runs.iloc[0]["run_id"])


def get_hist(run_id: str, key: str, max_step: int | None = None):
    h = CLIENT.get_metric_history(run_id, key)
    if max_step is not None:
        h = [m for m in h if m.step <= max_step]
    h = sorted(h, key=lambda x: x.step)
    if not h:
        return np.array([]), np.array([])
    return np.array([m.step for m in h], dtype=float), np.array([m.value for m in h], dtype=float)


def ema(y: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    if len(y) == 0:
        return y
    out = np.empty_like(y)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1 - alpha) * out[i - 1]
    return out


def render_png() -> tuple[bytes, dict]:
    r11 = get_latest_run("crazyflie_exp11_50k")
    r12 = get_latest_run("crazyflie_exp12_50k")
    r17 = get_latest_run("crazyflie_exp17_50k")
    r18 = get_latest_run("crazyflie_exp18_50k")
    r20 = get_latest_run("crazyflie_exp20_5k")
    r21 = get_latest_run("crazyflie_exp21_ft_5k")
    r22 = get_latest_run("crazyflie_exp22_50k")
    r24 = get_latest_run("crazyflie_exp24_50k")
    r25 = get_latest_run("crazyflie_exp25_50k")

    ref_run = r25 or r24 or r22
    s_ref, _ = get_hist(ref_run, "train/sim_reward") if ref_run else (np.array([]), np.array([]))
    max_step = int(s_ref[-1]) if len(s_ref) > 0 else 1000
    step_dur_s, step_dur_v = get_hist(ref_run, "train/step_duration_s") if ref_run else (np.array([]), np.array([]))
    sec_per_iter = float(np.median(step_dur_v[-20:])) if len(step_dur_v) > 0 else 0.75
    target_iters = 50000
    eta_min = max(0.0, (target_iters - max_step) * sec_per_iter / 60.0)

    series = [(EXP4_ID, "#1f77b4", "exp4 (Nominal, 256x256x128, 512^3 WM)")]
    if r11:
        series.append((r11, "#ff7f0e", "exp11 (Markovian ctx, RR=12, full spin)"))
    if r12:
        series.append((r12, "#2ca02c", "exp12 (Markovian ctx, RR=12, yaw-only anti-spin)"))
    if r17:
        series.append((r17, "#d62728", "exp17 (hist=4, spin=-0.012, tilt=1.5, 50k)"))
    if r24:
        series.append((r24, "#9467bd", "exp24 (Clean-yaw, old DR, 25k)"))
    if r25:
        series.append((r25, "#17becf", f"exp25 (Clean-yaw + reduced DR/noise @{max_step})"))

    metrics = [
        ("train/sim_reward", "Sim Reward (per step)", False),
        ("train/bptt_reward", "BPTT Reward", False),
        ("train/wm_loss", "World Model Total Loss", True),
        ("raw/wm/Loss/root_pos", "WM Loss: root_pos", True),
        ("raw/wm/Loss/root_ori", "WM Loss: root_ori", True),
        ("raw/policy/Policy/grad_norm", "Policy Grad Norm", True),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 11.5), dpi=120)
    axes = axes.flatten()
    latest = {}

    for ax, (mkey, title, use_log) in zip(axes, metrics):
        all_pos = True
        row = {}
        for rid, color, label in series:
            s, v = get_hist(rid, mkey, max_step=max_step)
            if len(s) == 0:
                continue
            if np.any(v <= 0):
                all_pos = False
            ax.plot(s, v, color=color, alpha=0.20, linewidth=0.9)
            ax.plot(s, ema(v), color=color, linewidth=2.1, label=label)
            row[label.split(" ")[0]] = float(v[-1])
        latest[mkey] = row
        if use_log and all_pos:
            ax.set_yscale("log")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(f"Iteration (0 - {max_step})")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"Live Loss & Reward Dynamics (0–{max_step} / 50,000) | {sec_per_iter:.2f}s/iter | ETA ~{eta_min:.0f} min",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue(), {"step": max_step, "sec_per_iter": sec_per_iter, "eta_min": eta_min, "latest": latest}


_CACHE_LOCK = None
_CACHE_PNG: bytes = b""
_CACHE_INFO: dict = {"step": 0, "sec_per_iter": 0.0, "eta_min": 0.0, "latest": {}}


def _refresh_loop():
    global _CACHE_PNG, _CACHE_INFO
    while True:
        try:
            png, info = render_png()
            with _CACHE_LOCK:
                _CACHE_PNG = png
                _CACHE_INFO = info
        except Exception:
            pass
        time.sleep(10)


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        with _CACHE_LOCK:
            png = _CACHE_PNG
            info = _CACHE_INFO
        if self.path.startswith("/plot.png"):
            if not png:
                png, info = render_png()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(png)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(png)
            return

        pct = 100.0 * info["step"] / 50000.0
        html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="15">
  <title>SuperMimic Live Training Dashboard</title>
  <style>
    body {{ font-family: system-ui, sans-serif; background: #111; color: #eee; margin: 16px auto; max-width: 1200px; }}
    .bar {{ background: #222; border-radius: 6px; padding: 10px 14px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; }}
    img {{ width: 100%; border-radius: 8px; background: #fff; }}
  </style>
</head>
<body>
  <div class="bar">
    <div><strong>exp10</strong>: {info['step']:,} / 50,000 ({pct:.1f}%) &nbsp;|&nbsp; {info['sec_per_iter']:.2f}s/iter &nbsp;|&nbsp; ETA ~{info['eta_min']:.0f} min</div>
    <div style="color:#aaa;font-size:13px;">Auto-refreshes every 15s &nbsp;|&nbsp; Updated {time.strftime('%H:%M:%S')}</div>
  </div>
  <img src="/plot.png?t={int(time.time())}" alt="Live Loss and Reward Dynamics">
</body>
</html>"""
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    import threading
    _CACHE_LOCK = threading.Lock()
    _CACHE_PNG, _CACHE_INFO = render_png()
    t = threading.Thread(target=_refresh_loop, daemon=True)
    t.start()
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("0.0.0.0", 5001), Handler) as httpd:
        print("Live dashboard serving on http://0.0.0.0:5001", flush=True)
        httpd.serve_forever()
