#!/usr/bin/env python3
"""Local web GUI to build <ds>/episode_offsets.csv conveniently.

Lists every episode in <ds>/body_data with its duration, lets you scrub each
episode's camera video to mark the head/tail trim, drop episodes you don't want,
preview the CSV live, and save it straight to <ds>/episode_offsets.csv.

The CSV the pipeline reads has exactly: orig_id, head_s, tail_s (one row per KEPT
episode; row order = dataset order; head/tail are seconds trimmed off start/end).

Run (any env with h5py -- e.g. humandata):
    python offsets_gui.py --ds /home/chen/Datasets/0716 [--port 8000]
then open the printed URL. Stdlib only; no pip installs.
"""
import argparse
import csv
import glob
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import http.server
import socketserver

import h5py
import numpy as np

DS = None
CACHE = None          # dir for browser-playable (H.264) transcodes
FFMPEG = shutil.which("ffmpeg") or "/home/chen/miniconda3/bin/ffmpeg"
FFPROBE = shutil.which("ffprobe") or "/home/chen/miniconda3/bin/ffprobe"
_DUR_CACHE = {}


def _mp4_duration(path):
    try:
        r = subprocess.run(
            [FFPROBE, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "format=duration", "-of", "csv=p=0", path],
            capture_output=True, text=True)
        return float(r.stdout.strip())
    except Exception:
        return None


def _frame_times(oid):
    """Real per-frame times (seconds from clip start), from the camera_ts sidecar.

    Prefer the camera's hardware clock (camera_ts_ns) -- it reflects the true capture
    cadence, which is what makes motion look right -- else the PC-receipt clock.
    Returns a 1-D float array or None.
    """
    p = f"{DS}/body_data/episode_{oid}_camera_ts.npz"
    if not os.path.exists(p):
        return None
    try:
        z = np.load(p)
        for key in ("camera_ts_ns", "local_ts_ns"):
            if key in z:
                t = np.asarray(z[key], np.int64)
                if t.shape[0] > 1:
                    return (t - int(t[0])) / 1e9
    except Exception:
        pass
    return None


def _plain_transcode(src, tmp):
    r = subprocess.run(
        [FFMPEG, "-y", "-i", src, "-c:v", "libx264", "-preset", "veryfast",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", "-f", "mp4", tmp],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return r.returncode == 0 and os.path.exists(tmp)


def _retime_perframe(src, times, tmp):
    """Rebuild the clip so each frame is held for its real inter-frame gap.

    The camera's rate varies *within* an episode (auto-exposure), so a single stretch
    factor can't fix it -- some parts end up too slow, others too fast. Extract the
    frames and re-mux via the concat demuxer with per-frame durations from the real
    timestamps, giving a VFR clip whose playback matches true capture timing.
    """
    work = tempfile.mkdtemp(dir=CACHE, prefix="rt_")
    try:
        ex = subprocess.run(
            [FFMPEG, "-y", "-i", src, "-qscale:v", "2", f"{work}/f%06d.jpg"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if ex.returncode != 0:
            return False
        frames = sorted(glob.glob(f"{work}/f*.jpg"))
        n = min(len(frames), len(times))
        if n < 2:
            return False
        gaps = np.diff(times[:n])
        fallback = float(np.median(gaps)) if gaps.size else 1.0 / 30
        listfile = f"{work}/list.txt"
        with open(listfile, "w") as f:
            f.write("ffconcat version 1.0\n")
            for k in range(n):
                d = float(gaps[k]) if k < n - 1 else fallback
                f.write(f"file '{frames[k]}'\n")
                f.write(f"duration {max(d, 1e-3):.6f}\n")
            f.write(f"file '{frames[n - 1]}'\n")  # concat holds last frame via a repeat
        # passthrough (not vfr) keeps every frame; a fine timescale stops the small
        # gaps (~18ms) from colliding and dropping frames -> tight PTS alignment.
        r = subprocess.run(
            [FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", listfile,
             "-fps_mode", "passthrough", "-video_track_timescale", "90000",
             "-c:v", "libx264", "-preset", "veryfast",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-f", "mp4", tmp],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return r.returncode == 0 and os.path.exists(tmp)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def playable_video(oid):
    """Return a browser-playable, real-time H.264 copy of episode <oid>'s left eye.

    Fixes vs the raw file: (1) transcode mp4v -> H.264, which HTML5 <video> can decode;
    (2) if the camera's real rate differed from the nominal 30fps, re-time each frame
    from its camera_ts timestamp so playback matches true capture timing and video time
    maps to episode time (so trim marks are correct). Cached.
    """
    src = f"{DS}/body_data/episode_{oid}_left.mp4"
    if not os.path.exists(src):
        return None
    dst = f"{CACHE}/episode_{oid}.mp4"
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        return dst
    os.makedirs(CACHE, exist_ok=True)
    tmp = f"{dst}.tmp"

    times = _frame_times(oid)
    nominal = _mp4_duration(src)
    ok = False
    # Per-frame re-time only when real duration disagrees with nominal (dim-light clips).
    if times is not None and nominal and abs(float(times[-1]) - nominal) > 0.3:
        ok = _retime_perframe(src, times, tmp)
    if not ok:
        ok = _plain_transcode(src, tmp)
    if not ok:
        if os.path.exists(tmp):
            os.remove(tmp)
        return src  # fall back to original (may not play, but won't 500)
    os.replace(tmp, dst)
    return dst


def episode_duration(oid):
    if oid in _DUR_CACHE:
        return _DUR_CACHE[oid]
    dur = None
    try:
        with h5py.File(f"{DS}/body_data/episode_{oid}.hdf5", "r") as f:
            if "local_timestamps_ns" in f and f["local_timestamps_ns"].shape[0] > 1:
                ts = f["local_timestamps_ns"]
                dur = (int(ts[-1]) - int(ts[0])) / 1e9
            elif "body_pose" in f:
                fps = round(1.0 / float(f.attrs.get("collection_interval_s", 0.01)))
                dur = f["body_pose"].shape[0] / fps
    except Exception:
        dur = None
    _DUR_CACHE[oid] = dur
    return dur


def list_episodes():
    out = []
    d = f"{DS}/body_data"
    for fn in os.listdir(d):
        m = re.match(r"episode_(\d+)\.hdf5$", fn)
        if not m:
            continue
        oid = int(m.group(1))
        out.append({
            "id": oid,
            "duration": episode_duration(oid),
            "video": os.path.exists(f"{d}/episode_{oid}_left.mp4"),
        })
    out.sort(key=lambda e: e["id"])
    return out


def load_csv():
    p = f"{DS}/episode_offsets.csv"
    rows = {}
    order = []
    if os.path.exists(p):
        with open(p) as f:
            for r in csv.DictReader(f):
                try:
                    oid = int(r["orig_id"])
                    rows[oid] = {"head": float(r["head_s"]), "tail": float(r["tail_s"])}
                    order.append(oid)
                except (KeyError, ValueError):
                    continue
    return {"rows": rows, "order": order}


def save_csv(items):
    """items: [{id, head, tail}, ...] already in the desired dataset order."""
    p = f"{DS}/episode_offsets.csv"
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["orig_id", "head_s", "tail_s"])
    for it in items:
        w.writerow([int(it["id"]), float(it["head"]), float(it["tail"])])
    text = buf.getvalue()
    with open(p, "w", newline="") as f:
        f.write(text)
    return p, text


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>episode_offsets.csv builder</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.45 system-ui,sans-serif; background:#f6f7f9; color:#1a1a1a; }
  @media (prefers-color-scheme: dark){ body{ background:#15171a; color:#e8e8e8; } }
  header { padding:10px 16px; background:#2d6cdf; color:#fff; display:flex; gap:16px; align-items:baseline; flex-wrap:wrap; }
  header b { font-size:16px; } header .ds { opacity:.9; font-family:ui-monospace,monospace; }
  .wrap { display:flex; gap:16px; padding:16px; align-items:flex-start; }
  .left { flex:1 1 560px; min-width:420px; }
  .right { flex:0 0 460px; position:sticky; top:16px; }
  .card { background:#fff; border:1px solid #dfe3ea; border-radius:10px; padding:12px; }
  @media (prefers-color-scheme: dark){ .card{ background:#1e2126; border-color:#333; } }
  table { border-collapse:collapse; width:100%; }
  th,td { padding:6px 8px; text-align:left; border-bottom:1px solid #eceef2; white-space:nowrap; }
  @media (prefers-color-scheme: dark){ th,td{ border-color:#2a2d33; } }
  th { position:sticky; top:0; background:inherit; font-weight:600; }
  tr.sel td { background:#eaf1ff; } @media (prefers-color-scheme: dark){ tr.sel td{ background:#22314f; } }
  tr.excluded td { opacity:.42; }
  .tablebox { max-height:70vh; overflow:auto; }
  input[type=number] { width:70px; padding:3px 5px; border:1px solid #cfd4dc; border-radius:6px; background:transparent; color:inherit; }
  input.bad { border-color:#e5484d; background:#fdecec; }
  button { cursor:pointer; border:1px solid #cfd4dc; background:#fff; border-radius:7px; padding:6px 11px; color:inherit; font:inherit; }
  @media (prefers-color-scheme: dark){ button{ background:#2a2e35; border-color:#3a3f47; } input{ border-color:#3a3f47; } }
  button.primary { background:#2d6cdf; color:#fff; border-color:#2d6cdf; }
  button.small { padding:3px 8px; font-size:12px; }
  .row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin:8px 0; }
  video { width:100%; border-radius:8px; background:#000; }
  .kept { font-variant-numeric:tabular-nums; }
  pre { background:#0d1117; color:#c9d1d9; padding:10px; border-radius:8px; overflow:auto; max-height:220px; margin:0; }
  .muted { opacity:.6; } .warn { color:#e5484d; } .ok { color:#2ea043; }
  #status { min-height:1.2em; }
  .marks { display:flex; gap:6px; font-family:ui-monospace,monospace; }
</style></head><body>
<header>
  <b>episode_offsets.csv builder</b>
  <span class="ds" id="dsPath"></span>
  <span id="counts" class="muted"></span>
</header>
<div class="wrap">
  <div class="left">
    <div class="card">
      <div class="row">
        <button class="small" onclick="setAll(true)">include all</button>
        <button class="small" onclick="setAll(false)">exclude all</button>
        <span class="muted">|</span>
        <label>apply head <input type="number" id="bulkHead" step="0.1" value="0" style="width:60px"></label>
        <label>tail <input type="number" id="bulkTail" step="0.1" value="0" style="width:60px"></label>
        <button class="small" onclick="applyBulk()">to included</button>
      </div>
      <div class="tablebox">
        <table>
          <thead><tr><th>keep</th><th>id</th><th>dur (s)</th><th>head_s</th><th>tail_s</th><th>kept</th><th></th></tr></thead>
          <tbody id="rows"></tbody>
        </table>
      </div>
    </div>
  </div>
  <div class="right">
    <div class="card">
      <div id="vidTitle" class="muted">click an episode to preview</div>
      <video id="vid" controls preload="metadata"></video>
      <div class="row">
        <button onclick="mark('head')">⟤ set head = now</button>
        <button onclick="mark('tail')">set tail = now ⟥</button>
        <span id="vtime" class="kept muted"></span>
      </div>
      <div class="marks muted" id="markInfo"></div>
      <p class="muted" style="font-size:12px">Video is the camera view (approx. episode time). Head = seconds into the clip where the motion should start; tail = seconds trimmed off the end.</p>
    </div>
    <div class="card" style="margin-top:12px">
      <div class="row">
        <button class="primary" onclick="save()">Save CSV</button>
        <button onclick="load()">Reload from disk</button>
        <span id="status"></span>
      </div>
      <pre id="preview"></pre>
    </div>
  </div>
</div>
<script>
let eps = [];        // [{id,duration,video}]
let st = {};         // id -> {included, head, tail}
let selId = null;

async function load() {
  const [e, c] = await Promise.all([
    fetch('api/episodes').then(r=>r.json()),
    fetch('api/csv').then(r=>r.json())
  ]);
  eps = e.episodes;
  document.getElementById('dsPath').textContent = e.ds;
  const hasCsv = c.order.length > 0;
  st = {};
  for (const ep of eps) {
    const row = c.rows[ep.id];
    st[ep.id] = row
      ? { included:true, head:row.head, tail:row.tail }
      : { included: !hasCsv, head:0, tail:0 };   // no CSV -> default include all
  }
  render();
  setStatus(hasCsv ? 'loaded existing episode_offsets.csv' : 'no CSV yet — all episodes included by default', 'ok');
}

function render() {
  const tb = document.getElementById('rows');
  tb.innerHTML = '';
  for (const ep of eps) {
    const s = st[ep.id];
    const tr = document.createElement('tr');
    tr.className = (ep.id===selId?'sel ':'') + (s.included?'':'excluded');
    const dur = ep.duration==null ? '?' : ep.duration.toFixed(1);
    tr.innerHTML = `
      <td><input type="checkbox" ${s.included?'checked':''} onchange="tog(${ep.id},this.checked)"></td>
      <td>${ep.id}</td>
      <td class="muted">${dur}</td>
      <td><input type="number" step="0.1" min="0" value="${s.head}" onchange="setv(${ep.id},'head',this.value)"></td>
      <td><input type="number" step="0.1" min="0" value="${s.tail}" onchange="setv(${ep.id},'tail',this.value)"></td>
      <td class="kept" id="kept${ep.id}"></td>
      <td>${ep.video?`<button class="small" onclick="pick(${ep.id})">▶ video</button>`:'<span class="muted">no vid</span>'}</td>`;
    tb.appendChild(tr);
    updKept(ep.id);
  }
  document.getElementById('counts').textContent =
    `${eps.length} episodes · ${Object.values(st).filter(s=>s.included).length} kept`;
  preview();
}

function updKept(id){
  const ep = eps.find(e=>e.id===id), s = st[id];
  const cell = document.getElementById('kept'+id);
  if(!cell) return;
  if(ep.duration==null){ cell.textContent='?'; return; }
  const k = ep.duration - (+s.head) - (+s.tail);
  cell.textContent = k.toFixed(1);
  cell.className = 'kept' + (k<=0?' warn':'');
}

function tog(id,v){ st[id].included=v; render(); }
function setv(id,f,v){ st[id][f]=parseFloat(v)||0; updKept(id); document.getElementById('counts');
  render(); }
function setAll(v){ for(const id in st) st[id].included=v; render(); }
function applyBulk(){
  const h=parseFloat(document.getElementById('bulkHead').value)||0;
  const t=parseFloat(document.getElementById('bulkTail').value)||0;
  for(const id in st) if(st[id].included){ st[id].head=h; st[id].tail=t; }
  render();
}

function pick(id){
  selId=id;
  const v=document.getElementById('vid');
  v.src='video/'+id;
  document.getElementById('vidTitle').textContent='episode '+id;
  render();
  updMarks();
}
function mark(which){
  if(selId==null) return;
  const v=document.getElementById('vid'), ep=eps.find(e=>e.id===selId);
  const t=v.currentTime;
  if(which==='head'){ st[selId].head=Math.round(t*10)/10; }
  else { const d = ep.duration!=null?ep.duration:(v.duration||t); st[selId].tail=Math.max(0,Math.round((d-t)*10)/10); }
  render(); updMarks();
}
function updMarks(){
  if(selId==null) return;
  const s=st[selId], ep=eps.find(e=>e.id===selId);
  document.getElementById('markInfo').textContent =
    `head ${s.head}s  ·  tail ${s.tail}s  ·  kept window [${(+s.head).toFixed(1)}, ${ep.duration!=null?(ep.duration-s.tail).toFixed(1):'?'}]s`;
}
const vid=document.getElementById('vid');
vid.addEventListener('timeupdate',()=>{ document.getElementById('vtime').textContent = vid.currentTime.toFixed(2)+'s / '+(vid.duration||0).toFixed(1)+'s'; });

function orderedIncluded(){
  return eps.filter(e=>st[e.id].included).map(e=>({id:e.id, head:+st[e.id].head, tail:+st[e.id].tail}));
}
function preview(){
  const items=orderedIncluded();
  let txt='orig_id,head_s,tail_s\n'+items.map(i=>`${i.id},${i.head},${i.tail}`).join('\n');
  document.getElementById('preview').textContent=txt;
}
function setStatus(m,cls){ const s=document.getElementById('status'); s.textContent=m; s.className=cls||''; }

async function save(){
  const items=orderedIncluded();
  const bad=items.filter(i=>{const ep=eps.find(e=>e.id===i.id); return ep.duration!=null && ep.duration-i.head-i.tail<=0;});
  if(bad.length){ if(!confirm(`${bad.length} episode(s) have non-positive kept length (ids: ${bad.map(b=>b.id).join(',')}). Save anyway?`)) return; }
  setStatus('saving…');
  const r=await fetch('api/csv',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({items})});
  const j=await r.json();
  if(j.ok) setStatus('saved '+j.path+' ('+items.length+' rows)','ok'); else setStatus('error: '+j.error,'warn');
}
load();
</script></body></html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/" or path == "/index.html":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/episodes":
            self._json({"ds": DS, "episodes": list_episodes()})
        elif path == "/api/csv":
            self._json(load_csv())
        elif path.startswith("/video/"):
            self._serve_video(path[len("/video/"):])
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/csv":
            n = int(self.headers.get("Content-Length", 0))
            try:
                data = json.loads(self.rfile.read(n) or b"{}")
                path, _ = save_csv(data.get("items", []))
                self._json({"ok": True, "path": path})
            except Exception as e:
                self._json({"ok": False, "error": str(e)}, 400)
        else:
            self.send_error(404)

    def _serve_video(self, oid):
        if not re.match(r"^\d+$", oid):
            self.send_error(400); return
        path = playable_video(oid)  # H.264 transcode (cached); browsers can't play mp4v
        if not path or not os.path.exists(path):
            self.send_error(404); return
        size = os.path.getsize(path)
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng:
            m = re.match(r"bytes=(\d+)-(\d*)", rng)
            if m:
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
        end = min(end, size - 1)
        length = end - start + 1
        self.send_response(206 if rng else 200)
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        try:
            with open(path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def main():
    global DS, CACHE
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, help="dataset workspace dir (contains body_data/)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    DS = os.path.abspath(a.ds)
    if not os.path.isdir(f"{DS}/body_data"):
        raise SystemExit(f"no body_data/ under {DS}")
    CACHE = f"{DS}/.offsets_video_cache"  # H.264 transcodes for browser playback (safe to delete)
    if not (shutil.which("ffmpeg") or os.path.exists(FFMPEG)):
        print("WARNING: ffmpeg not found; videos (mp4v) won't play in the browser")
    srv = Server((a.host, a.port), Handler)
    print(f"episode_offsets.csv builder for {DS}")
    print(f"  open  http://{a.host}:{a.port}/")
    print("  Ctrl-C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
