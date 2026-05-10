"""
FITS — Integrated App
Task 1 (Body & Face Analysis) → Task 2 (Fashion Recommendations)
Complete end-to-end Streamlit flow.

Run from the FITS root:
    streamlit run fits_app.py
"""

import os, sys, json, cv2, numpy as np, io, base64, uuid, hashlib, random, time, re, subprocess, shutil, shlex
import math as _math
import streamlit as st
import streamlit.components.v1 as _st_components
import pandas as pd
import torch
import requests
import warnings
warnings.filterwarnings("ignore", message=".*__path__.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*zoedepth.*", category=UserWarning)
from sentence_transformers import SentenceTransformer, util
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# ─── Path setup ───────────────────────────────────────────────────────────────
ROOT_DIR    = Path(r"C:\Users\tcq26\Downloads\FITS\FITS")
TASK1_DIR   = ROOT_DIR / "TASK_1"
TASK2_DIR   = ROOT_DIR / "TASK_2" / "FITS_Recommender" / "FITS_Recommender"
MODELS_DIR  = TASK1_DIR / "models"
CAPTURES_DIR = TASK1_DIR / "captures"
CAPTURES_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(TASK1_DIR))

from analysis import (analyze_view, fuse_results, draw_body_silhouettes,
                      draw_face_landmarks, rgb_to_hex, hex_to_rgb)
from db.json_repository import get_repository
import importlib as _importlib
import db.experiment_repository as _exp_repo_mod
_importlib.reload(_exp_repo_mod)
from db.experiment_repository import get_experiment_repository

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Task 2 absolute paths
PRODUCTS_CSV = str(TASK2_DIR / "products.csv")
IMAGES_ROOT  = str(TASK2_DIR / "all_products")
RULES_FILE   = str(TASK2_DIR / "fashion_rules.json")
SESSION_LOG  = str(TASK2_DIR / "session_logs.json")
USERS_DIR    = str(TASK2_DIR / "users")

# ── Timer-capture JS component ─────────────────────────────────────────────
_COMP_DIR = ROOT_DIR / "_components" / "timer_capture"

# HTML/JS component: shows live camera, configurable countdown, auto-grabs last frame,
# returns JPEG base64 to Python via Streamlit component messaging — no button needed.
# Timer duration is passed via component args: comp(seconds=3) or comp(seconds=10).
_TIMER_CAP_HTML = """\
<!DOCTYPE html><html><head><meta charset="utf-8"><style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#111;color:#eee;font-family:system-ui,sans-serif;padding:10px}
#wrap{position:relative;width:100%}
video{width:100%;border-radius:8px;display:block;background:#000;min-height:180px}
#snap{width:100%;border-radius:8px;display:none;height:auto;background:#000}
#ovr{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;pointer-events:none}
#cd{font-size:7rem;font-weight:800;display:none;line-height:1}
#bar{height:6px;border-radius:3px;margin:8px 0;overflow:hidden}
#fill{height:100%;width:0%;transition:width 1s linear;border-radius:3px}
#msg{text-align:center;font-size:.85rem;min-height:22px;padding:4px 0 8px}
.btn{width:100%;padding:10px;border:none;border-radius:6px;cursor:pointer;font-size:.95rem;font-weight:600;margin-top:6px;display:block}
#startBtn:disabled{cursor:default}
#retryBtn{display:none}
#captD{display:none;text-align:center;color:#4caf50;font-weight:700;padding:8px 0}
</style></head><body>
<div id="wrap">
  <video id="vid" autoplay playsinline muted></video>
  <canvas id="snap"></canvas>
  <div id="ovr"><div id="cd"></div></div>
</div>
<div id="bar"><div id="fill"></div></div>
<div id="msg">Tap Start &#8212; get into position before the timer ends.</div>
<div id="captD">&#x2705; Photo captured! Scroll down to continue, or retry.</div>
<button class="btn" id="startBtn">&#x23F1; Start Timer</button>
<button class="btn" id="retryBtn">&#x1F501; Try Again</button>
<script>
(function(){
  var SECONDS=10;

  function post(type,payload){
    window.parent.postMessage(Object.assign({isStreamlitMessage:true,type},payload),"*");
  }
  var ST={
    ready:function(){post("streamlit:componentReady",{apiVersion:1});},
    setValue:function(v){post("streamlit:setComponentValue",{value:v,dataType:"json"});},
    height:function(h){post("streamlit:setFrameHeight",{height:h});}
  };
  var vid=document.getElementById("vid");
  var snap=document.getElementById("snap");
  var cd=document.getElementById("cd");
  var fill=document.getElementById("fill");
  var msg=document.getElementById("msg");
  var startBtn=document.getElementById("startBtn");
  var retryBtn=document.getElementById("retryBtn");
  var captD=document.getElementById("captD");
  var stream=null;
  var running=false;
  var msgHint="Tap Start \u2014 get into position before the timer ends.";

  function applyTheme(t){
    var isL=t==="light";
        var isC=t==="comfort";
    document.body.style.background=isL?"#f4f0ff":"#10131d";
    document.body.style.color=isL?"#1a1033":"#eef6ff";
    cd.style.color=isL?"#6b21ff":"#2fe5e5";
    cd.style.textShadow=isL?"0 0 34px rgba(107,33,255,.95)":"0 0 34px rgba(47,229,229,.95)";
    document.getElementById("bar").style.background=isL?"#ddd1ff":"#2a3144";
    fill.style.background=isL?"#6b21ff":"#2fe5e5";
    msg.style.color=isL?"#5f4b9f":"#b9d5ef";
    startBtn.style.background=isL?"#6b21ff":"#2fe5e5";
    startBtn.style.color=isL?"#fff":"#111";
    retryBtn.style.background=isL?"#ddd1ff":"#30384e";
    retryBtn.style.color=isL?"#4b4466":"#e7f3ff";
        if(isC){
            document.body.style.background="#121315";
            document.body.style.color="#e5e2da";
            cd.style.color="#b2d878";
            cd.style.textShadow="0 0 24px rgba(178,216,120,.52)";
            document.getElementById("bar").style.background="#2a2d33";
            fill.style.background="#b2d878";
            msg.style.color="#c9d8a7";
            startBtn.style.background="#b2d878";
            startBtn.style.color="#131416";
            retryBtn.style.background="#2d3035";
            retryBtn.style.color="#d6d1c5";
        }
  }
  applyTheme("dark");
  window.addEventListener("message",function(e){
    if(e.data&&e.data.type==="streamlit:render"){
      if(e.data.args&&e.data.args.seconds){SECONDS=parseInt(e.data.args.seconds,10)||10;}
      if(e.data.args&&e.data.args.hint){msgHint=e.data.args.hint;msg.textContent=msgHint;}
      if(e.data.args&&e.data.args.theme){applyTheme(e.data.args.theme);}
      startBtn.textContent="\u23F1 Start "+SECONDS+"-second Timer";
      startCam();
    }
  });
  ST.ready();
  ST.height(320);

  function uh(){setTimeout(function(){ST.height(document.body.scrollHeight+24);},250);}

  function startCam(){
    if(stream)return;
    navigator.mediaDevices.getUserMedia({video:{facingMode:"user",width:{ideal:1080}},audio:false})
      .then(function(s){
        stream=s;
        vid.srcObject=s;
        uh();
      }).catch(function(e){
        msg.textContent="\u26a0 Camera access denied: "+e.message;
      });
  }

  startBtn.addEventListener("click",function(){
    if(running)return;
    running=true;
    startBtn.disabled=true;
    retryBtn.style.display="none";
    captD.style.display="none";
    snap.style.display="none";
    vid.style.display="block";
    tick(SECONDS);
    uh();
  });

  retryBtn.addEventListener("click",function(){
    running=false;
    startBtn.disabled=false;
    startBtn.style.display="block";
    retryBtn.style.display="none";
    captD.style.display="none";
    snap.style.display="none";
    vid.style.display="block";
    cd.style.display="none";
    fill.style.transition="none";
    fill.style.width="0%";
    setTimeout(function(){fill.style.transition="width 1s linear";},50);
    msg.textContent=msgHint;
    stream=null;
    startCam();
    uh();
  });

  function tick(n){
    if(n<0){grab();return;}
    cd.style.display=n>0?"block":"none";
    cd.textContent=n>0?n:"";
    fill.style.width=((SECONDS-n)/SECONDS*100)+"%";
    var half=Math.ceil(SECONDS/2);
    msg.textContent=n>0?(n>half?"Get into position\u2026":"Hold still\u2026"):"\U0001F4F8 Capturing\u2026";
    setTimeout(function(){tick(n-1);},1000);
  }

  function grab(){
    if(!vid.videoWidth){msg.textContent="\u26a0 No video signal. Please retry.";running=false;startBtn.disabled=false;startBtn.style.display="block";return;}
    snap.width=vid.videoWidth;
    snap.height=vid.videoHeight;
    snap.getContext("2d").drawImage(vid,0,0);
    stream.getTracks().forEach(function(t){t.stop();});
    vid.style.display="none";
    snap.style.display="block";
    cd.style.display="none";
    captD.style.display="block";
    retryBtn.style.display="block";
    startBtn.style.display="none";
    msg.textContent="";
    fill.style.width="100%";
    var b64=snap.toDataURL("image/jpeg",0.92).split(",")[1];
    ST.setValue(b64);
    uh();
  }

  startCam();
})();
</script></body></html>
"""


# Write the component file at import time (no cache — cheap IO, always fresh)
_COMP_DIR.mkdir(parents=True, exist_ok=True)
(_COMP_DIR / "index.html").write_text(_TIMER_CAP_HTML, encoding="utf-8")

# Declare the component once at module level
_timer_capture_comp = _st_components.declare_component(
    "timer_capture", path=str(_COMP_DIR)
)


def _timer_capture_component(cam_key: str, seconds: int = 10, hint: str = ""):
    """Call the timer-capture JS component. Returns JPEG bytes or None."""
    # Each retry gets a fresh key so the component remounts cleanly
    retry_k = f"_cam_retry_{cam_key}"
    if retry_k not in st.session_state:
        st.session_state[retry_k] = 0
    _theme = st.session_state.get("theme", "dark")
    b64 = _timer_capture_comp(key=f"tc_{cam_key}_{st.session_state[retry_k]}", seconds=seconds, hint=hint, theme=_theme)
    if b64 and isinstance(b64, str) and len(b64) > 100:
        st.session_state[retry_k] += 1
        return io.BytesIO(base64.b64decode(b64))
    return None
# TASK 1 — MediaPipe model loader
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def _load_models_image():
    base_opts = mp_python.BaseOptions
    fl_opts = mp_vision.FaceLandmarkerOptions(
        base_options=base_opts(model_asset_path=str(MODELS_DIR / "face_landmarker.task")),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1, output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        min_face_detection_confidence=0.4,
        min_face_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    pl_opts = mp_vision.PoseLandmarkerOptions(
        base_options=base_opts(model_asset_path=str(MODELS_DIR / "pose_landmarker_heavy.task")),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1, output_segmentation_masks=False,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    seg_opts = mp_vision.ImageSegmenterOptions(
        base_options=base_opts(model_asset_path=str(MODELS_DIR / "selfie_segmentation.tflite")),
        running_mode=mp_vision.RunningMode.IMAGE,
        output_category_mask=True, output_confidence_masks=False,
    )
    return (
        mp_vision.FaceLandmarker.create_from_options(fl_opts),
        mp_vision.PoseLandmarker.create_from_options(pl_opts),
        mp_vision.ImageSegmenter.create_from_options(seg_opts),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 1 — Positioning HUD & helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_position_guide(view_key: str, light_mode: bool = False):
    if light_mode:
        BG    = "#f4f0ff"; GOLD  = "#7c3aed"; DIM = "#c4b5fd"
        LABEL = "#9c88cc"; TEAL  = "#7c3aed"; BC  = "#c4b5fd"
    else:
        BG    = "#111111"; GOLD  = "#c8b89a"; DIM = "#555555"
        LABEL = "#888888"; TEAL  = "#6cc8c8"; BC  = "#444444"
    fig, ax = plt.subplots(figsize=(1.9, 3.2))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.set_xlim(0, 180); ax.set_ylim(310, 0)
    ax.set_aspect("equal"); ax.axis("off")
    oval = Ellipse((90, 78), width=92, height=120, fill=False, edgecolor=GOLD,
                   linewidth=1.8, linestyle=(0, (7, 4)))
    ax.add_patch(oval)
    ax.plot([90, 90], [4, 16], color=DIM, lw=0.9, ls=(0, (3, 2)))
    ax.text(93, 14, "crown", fontsize=5, color=LABEL, va="top")
    ax.plot([52, 128], [136, 136], color=GOLD, lw=0.9, ls=(0, (4, 3)), alpha=0.6)
    ax.text(130, 136, "chin", fontsize=5, color=LABEL, va="center")
    ax.plot([78, 74], [139, 188], color=GOLD, lw=1.6)
    ax.plot([102, 106], [139, 188], color=GOLD, lw=1.6)
    ax.plot([18, 162], [188, 188], color=GOLD, lw=2.2)
    ax.plot([18, 18], [181, 195], color=GOLD, lw=1.8)
    ax.plot([162, 162], [181, 195], color=GOLD, lw=1.8)
    ax.text(90, 207, "shoulders", fontsize=5.5, color=LABEL, ha="center", va="top")
    if view_key == "front":
        ax.plot([90, 90], [16, 136], color=DIM, lw=0.8, ls=(0, (2, 3)), alpha=0.6)
        ax.plot([90, 90], [142, 183], color=DIM, lw=0.8, ls=(0, (2, 3)), alpha=0.6)
    if view_key == "face":
        t1, t2 = ("Face close-up", "~0.5 m from camera")
        ax.text(90, 237, t1, fontsize=7, color=TEAL, ha="center", va="top", fontweight="bold")
        ax.text(90, 252, t2, fontsize=5.5, color=LABEL, ha="center", va="top")
        ax.text(90, 271, "Fill frame chin to forehead", fontsize=5, color=LABEL, ha="center", va="top")
        ax.text(90, 282, "No sunglasses, look straight ahead", fontsize=4.5, color=LABEL, ha="center", va="top")
        ax.text(90, 293, "Head inside the oval", fontsize=5, color=LABEL, ha="center", va="top")
    else:
        t1, t2 = ("Face camera directly", "Stand ~1.5 m away")
        ax.plot([168, 168], [18, 188], color=DIM, lw=0.8)
        ax.plot([162, 174], [18, 18], color=DIM, lw=0.8)
        ax.plot([162, 174], [188, 188], color=DIM, lw=0.8)
        ax.plot([165, 171], [103, 103], color=DIM, lw=0.8)
        ax.text(162, 107, "1.5 m", fontsize=4.5, color=LABEL, ha="right", va="center")
        ax.text(162, 115, "from lens", fontsize=4.0, color=LABEL, ha="right", va="center")
        for xs, ys in [([8,8,28],[28,8,8]),([152,172,172],[8,8,28]),
                       ([8,8,28],[200,220,220]),([152,172,172],[220,220,200])]:
            ax.plot(xs, ys, color=BC, lw=1.3)
        ax.text(90, 237, t1, fontsize=7, color=TEAL, ha="center", va="top", fontweight="bold")
        ax.text(90, 252, t2, fontsize=5.5, color=LABEL, ha="center", va="top")
        ax.text(90, 271, "Stand ~1.5 m from the camera", fontsize=5, color=LABEL, ha="center", va="top")
        ax.text(90, 282, "Full body in frame (also used for try-on)", fontsize=4.5, color=LABEL, ha="center", va="top")
        ax.text(90, 293, "Head inside the oval", fontsize=5, color=LABEL, ha="center", va="top")
    fig.tight_layout(pad=0.05)
    return fig


def _annotate_thumbnail(img_bgr: np.ndarray, view_key: str) -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]
    cx = w // 2; cy_h = int(h * 0.30)
    rx = int(w * 0.145); ry = int(h * 0.245)
    chin_y = cy_h + ry; sh_y = int(h * 0.73)
    sh_hw = int(w * 0.29); nk_hw = int(w * 0.048)
    GOLD = (154, 184, 200); DIM = (90, 90, 90); TEAL = (200, 195, 80)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy_h), (int(rx*1.3), int(ry*1.3)), 0, 0, 360, 255, -1)
    cv2.rectangle(mask, (cx - sh_hw, chin_y), (cx + sh_hw, sh_y + 20), 255, -1)
    inv = 255 - mask
    out[inv > 0] = (out[inv > 0].astype(np.float32) * 0.38).astype(np.uint8)
    pts = [(int(cx + rx * _math.cos(_math.radians(d))),
            int(cy_h + ry * _math.sin(_math.radians(d)))) for d in range(0, 362, 5)]
    for i in range(0, len(pts) - 1, 2):
        cv2.line(out, pts[i], pts[i+1], GOLD, 2, cv2.LINE_AA)
    cv2.line(out, (cx - rx, chin_y), (cx + rx, chin_y), DIM, 1)
    cv2.line(out, (cx - nk_hw, chin_y), (cx - nk_hw, sh_y), GOLD, 2, cv2.LINE_AA)
    cv2.line(out, (cx + nk_hw, chin_y), (cx + nk_hw, sh_y), GOLD, 2, cv2.LINE_AA)
    cv2.line(out, (cx - sh_hw, sh_y), (cx + sh_hw, sh_y), GOLD, 2, cv2.LINE_AA)
    for sx in (cx - sh_hw, cx + sh_hw):
        cv2.line(out, (sx, sh_y - 8), (sx, sh_y + 8), GOLD, 2, cv2.LINE_AA)
    cv2.putText(out, view_key.upper(), (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEAL, 2, cv2.LINE_AA)
    return out


def _swatch(hex_color: str, size: int = 40) -> str:
    _bdr = _theme_color("#e0d9f7", "#555", "#58554c")
    return (f'<div style="width:{size}px;height:{size}px;border-radius:6px;'
            f'background:{hex_color};border:1px solid {_bdr};display:inline-block;'
            f'vertical-align:middle;margin:0 4px"></div>')


def _theme_mode() -> str:
    return st.session_state.get("theme", "dark")


def _theme_color(light: str, dark: str, comfort: str | None = None) -> str:
    mode = _theme_mode()
    if mode == "light":
        return light
    if mode == "comfort":
        return comfort if comfort is not None else dark
    return dark


def _swatch_row_main(label: str, hex_color: str):
    if not hex_color or hex_color == "#000000":
        st.markdown(f"**{label}**: —")
        return
    r, g, b = hex_to_rgb(hex_color)
    lum = 0.299*r + 0.587*g + 0.114*b
    fg = "#000" if lum > 130 else "#fff"
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;margin:4px 0">'
        f'  <span style="min-width:90px;font-weight:600">{label}</span>'
        f'  {_swatch(hex_color, 36)}'
        f'  <span style="background:{hex_color};color:{fg};padding:2px 8px;'
        f'border-radius:4px;font-family:monospace;font-size:12px">{hex_color}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 2 — Recommender engine (extracted, path-aware)
# ═══════════════════════════════════════════════════════════════════════════════

class ConfigurableRecommender:
    @staticmethod
    def _parse_price(raw_value) -> float:
        """Best-effort numeric parsing for messy CSV price fields.

        Handles plain numbers, currency text, malformed multi-dot values, and ranges.
        Returns the first detected numeric token, or 0.0 when unavailable.
        """
        if raw_value is None:
            return 0.0

        s = str(raw_value).strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            return 0.0

        # Match first decimal/integer token (e.g., "35.7-59.5" -> "35.7").
        m = re.search(r"\d+(?:\.\d+)?", s)
        if m:
            try:
                return float(m.group(0))
            except Exception:
                pass

        # Final fallback: preserve only digits and dots, then keep first valid decimal part.
        cleaned = "".join(c for c in s if c.isdigit() or c == ".")
        if not cleaned:
            return 0.0
        first = cleaned.split(".")[0]
        rest = [p for p in cleaned.split(".")[1:] if p]
        candidate = first if not rest else f"{first}.{rest[0]}"
        try:
            return float(candidate)
        except Exception:
            return 0.0

    def __init__(self, products_csv, rules_file, images_root):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        # Keep rules_file path so it can be reloaded on every recommend call
        self.rules_file = rules_file
        self.rules = self._load_rules()
        df = pd.read_csv(products_csv)
        grouped = df.groupby("product_id")
        self.inventory = []

        # ── Pre-build a case-insensitive folder-name → absolute-path index ────
        # Scan all_products/ once so every per-product lookup is an O(1) dict hit.
        _img_exts = (".jpg", ".jpeg", ".png", ".webp")
        _folder_index = {}  # lower-cased folder name → absolute folder path
        if os.path.isdir(images_root):
            for _e in os.scandir(images_root):
                if _e.is_dir():
                    _folder_index[_e.name.lower()] = _e.path

        def _to_key(val) -> str:
            """Derive a folder-index lookup key from any column value.

            Handles:
              - bare product/folder names  (clean_name)
              - prefixed paths             (products/Folder Name)
              - cross-machine abs paths    (C:\\...\\products\\Folder Name\\img.jpg)
            Strategy: normalise separators, strip everything up to and including
            the first 'products/' segment, then take the first path component.
            """
            s = str(val or "").strip()
            if not s or s.lower() in {"nan", "none", "null"}:
                return ""
            norm = s.replace("\\", "/")
            idx = norm.lower().find("products/")
            if idx >= 0:
                norm = norm[idx + len("products/"):]
            return norm.split("/")[0].strip().lower()

        def _images_for_key(key: str) -> list:
            folder = _folder_index.get(key, "")
            if not folder:
                return []
            imgs = []
            try:
                for fname in sorted(os.listdir(folder)):
                    if fname.lower().endswith(_img_exts):
                        imgs.append(os.path.join(folder, fname))
            except OSError:
                pass
            return imgs

        for pid, g in grouped:
            pid_str = str(pid)
            first = g.iloc[0]
            image_paths = []

            # Try candidate keys in priority order; stop at the first folder hit.
            _seen_keys = set()
            for _row in g.itertuples(index=False):
                for _col in ("clean_name", "folder_path", "image_path", "folder_name", "name"):
                    _val = getattr(_row, _col, "") or ""
                    _k = _to_key(_val)
                    if not _k or _k in _seen_keys:
                        continue
                    _seen_keys.add(_k)
                    _imgs = _images_for_key(_k)
                    if _imgs:
                        image_paths = _imgs
                        break
                if image_paths:
                    break

            self.inventory.append({
                "id": pid_str,
                "name": str(first.get("clean_name", "")),
                "gender": str(first.get("gender", "")),
                "brand": str(first.get("brand", "")),
                "description": str(first.get("description", "")),
                "category": str(first.get("category_name", "")).strip().lower(),
                "pants_type": str(first.get("pants_type", "")).strip().lower(),
                "jeans_type": str(first.get("jeans_type", "")).strip().lower(),
                "dress_length": str(first.get("dress_length", "")).strip().lower(),
                "sleeve_length": str(first.get("sleeve_type", "")),
                "neck_type": str(first.get("neckline_type", "")),
                "color_tone": str(first.get("color", "")),
                "price": ConfigurableRecommender._parse_price(first.get("price", 0)),
                "fit": ConfigurableRecommender._derive_fit(
                    str(first.get("pants_type", "")).strip().lower(),
                    str(first.get("jeans_type", "")).strip().lower(),
                    str(first.get("description", "")),
                ),
                "images": image_paths,
            })

    def _load_rules(self) -> dict:
        """Read fashion_rules.json from disk. Called fresh on every recommend()."""
        try:
            with open(self.rules_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"⚠️ Could not load fashion_rules.json: {e}")
            return {}

    def _get_rule_block(self, trait_key, val_key):
        trait_rules = self.rules.get(trait_key)
        if not trait_rules:
            return None
        return trait_rules.get(val_key)

    def generate_query_and_filter(self, user_attributes, inventory_list):
        query_parts = []
        banned_items_indices = set()

        def _matches_avoid_term(text: str, term: str) -> bool:
            """Match avoid terms conservatively so generic words don't over-ban inventory.

            Example: the rule term "short" should not eliminate every "shorts" item.
            """
            text = str(text or "").lower()
            term = str(term or "").strip().lower()
            if not text or not term:
                return False
            if " " in term or "-" in term:
                return term in text
            return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) is not None

        for trait, value in user_attributes.items():
            if value is None or value == "":
                continue
            if trait.startswith("_"):
                continue  # internal metadata — handled separately
            trait_key = trait.lower()
            val_key = str(value).lower()
            rule_data = self._get_rule_block(trait_key, val_key)
            if rule_data is not None:
                if isinstance(rule_data, dict):
                    match_text = rule_data.get("match", "")
                    if match_text:
                        query_parts.append(match_text)
                    avoid_raw = rule_data.get("avoid", "")
                    avoid_terms = [t for t in avoid_raw.split() if t]
                    if avoid_terms:
                        for idx, item in enumerate(inventory_list):
                            desc_lower = (item.get("description","").lower()
                                          + " " + item.get("name","").lower())
                            if any(_matches_avoid_term(desc_lower, term) for term in avoid_terms):
                                banned_items_indices.add(idx)
                else:
                    query_parts.append(str(rule_data))
                continue
            query_parts.append(val_key)
        query_text = " ".join(q for q in query_parts if q.strip())
        try:
            gender = str(user_attributes.get("gender", "")).lower()
            cloth  = str(user_attributes.get("clothing_type", "")).lower()
        except Exception:
            gender = ""; cloth = ""
        if gender == "men" and cloth in {"top","tops","t_shirt","t_shirts"}:
            occ  = str(user_attributes.get("occasion", "")).lower()
            body = str(user_attributes.get("body_type", "")).lower()
            if occ == "sporty":
                query_parts.append("athletic performance breathable moisture-wicking sporty")
            elif occ == "event":
                query_parts.append("smart polished button-up dress shirt")
            elif occ == "casual":
                query_parts.append("casual graphic tee comfortable everyday")
            elif occ == "formals":
                query_parts.append("tailored dress shirt formal polished")
            if body == "inverted_triangle":
                query_parts.append("v-neck slim fit balanced shoulders")
            elif body == "rectangle":
                query_parts.append("structured layers tailored silhouette")
            elif body == "pear":
                query_parts.append("structured shoulders patterns details at top")
            query_text = " ".join(q for q in query_parts if q.strip())
        if gender == "women":
            occ = str(user_attributes.get("occasion", "")).lower()
            if cloth in {"shorts","short"}:
                if occ == "sporty":
                    query_parts.append("athletic running biker shorts gym breathable")
                else:
                    query_parts.append("denim shorts casual high waisted summer")
            if cloth in {"top","tops","t_shirt","t_shirts"}:
                if occ == "sporty":
                    query_parts.append("performance tee breathable athletic crop biker tank")
                elif occ == "event":
                    query_parts.append("blouse elegant dressy smart casual")
            query_text = " ".join(q for q in query_parts if q.strip())
        
        # ── Vibe hint from personality quiz ───────────────────────────────────
        vibe_hint = user_attributes.get("_vibe_hint", "")
        if vibe_hint:
            query_parts.append(vibe_hint)
            query_text = " ".join(q for q in query_parts if q.strip())

        return query_text, banned_items_indices

    @staticmethod
    def _derive_fit(pants_type: str, jeans_type: str, description: str) -> str:
        """Derive a normalised fit bucket from structured type fields + free-text description.

        Returned values are intentionally aligned with _FIT_PREF_INV in _step_recommendations():
          "loose"    → matches "Loose & Oversized" filter  ["loose","oversized","relaxed"]
          "oversized"→ matches "Loose & Oversized" filter
          "relaxed"  → matches "Loose & Oversized" and "Relaxed" filters
          "regular"  → matches "Relaxed" and "Regular" filters
          "slim"     → matches "Fitted" filter             ["fitted","slim","regular"]
          "fitted"   → matches "Fitted" filter
        """
        _LOOSE    = {"loose","wide_leg","flare","bootcut","barrel_leg","baggy","mom_fit"}
        _RELAXED  = {"relaxed","jogger","sweatpants","shorts"}
        _SLIM     = {"skinny","slim"}
        _REGULAR  = {"straight","regular","trouser","chino","khaki"}

        for val in (pants_type.lower().strip(), jeans_type.lower().strip()):
            if not val or val == "unknown":
                continue
            if val in _LOOSE:    return "loose"
            if val in _RELAXED:  return "relaxed"
            if val in _SLIM:     return "slim"
            if val in _REGULAR:  return "regular"

        # Fallback: scan description for fit signals
        desc = (description or "").lower()
        if any(w in desc for w in ("oversized", "over-sized", "over sized", "slouchy", "boxy")):
            return "oversized"
        if any(w in desc for w in ("loose fit", "loose-fit", "wide leg", "wide-leg", "baggy", "flare", "bootcut")):
            return "loose"
        if any(w in desc for w in ("relaxed fit", "relaxed-fit", "relaxed through", "slouch", "jogger")):
            return "relaxed"
        if any(w in desc for w in ("slim fit", "slim-fit", "skinny", "narrow leg")):
            return "slim"
        if any(w in desc for w in ("fitted", "bodycon", "body-hugging", "form-fitting", "tight")):
            return "fitted"
        if any(w in desc for w in ("straight leg", "straight-leg", "regular fit", "classic fit")):
            return "regular"
        return ""

    def _item_text(self, item):
        return " ".join([
            item.get("brand",""), item.get("name",""), item.get("description",""),
            f"category: {item.get('category','')}",
            f"pants type: {item.get('pants_type','')}",
            f"jeans type: {item.get('jeans_type','')}",
            f"dress length: {item.get('dress_length','')}",
            f"sleeve: {item.get('sleeve_length','')}",
            f"neck: {item.get('neck_type','')}",
            f"color tone: {item.get('color_tone','')}",
            f"fit: {item.get('fit','')}",
        ])

    def _map_clothing(self, cloth, gender):
        cloth = (cloth or "").strip().lower()
        gender = (gender or "").strip().lower()
        if not cloth:
            return None
        women_map = {
            "dress":    ["dress", "dresses"],
            "top":      ["top", "tops", "t_shirt", "t_shirts", "blouse", "shirt", "hoodie", "sweatshirt", "polo"],
            "trousers": ["pant", "pants", "jean", "jeans", "trouser"],
            "shorts":   ["shorts"],
            "skirt":    ["skirt", "skirts"],
        }
        men_map = {
            "top":      ["top", "tops", "t_shirt", "t_shirts", "shirt", "hoodie", "sweatshirt", "polo"],
            "trousers": ["pant", "pants", "jean", "jeans", "trouser"],
            "shorts":   ["shorts"],
        }
        if gender == "women":
            return women_map.get(cloth, [cloth])
        if gender == "men":
            return men_map.get(cloth, [cloth])
        return (women_map.get(cloth) or men_map.get(cloth) or [cloth])

    def recommend(self, user_attributes, clothing_type=None, top_k=5, random_seed=None):
        # Rules are loaded once at init; only reload if the file has changed
        try:
            _mtime = os.path.getmtime(self.rules_file)
            if not hasattr(self, '_rules_mtime') or self._rules_mtime != _mtime:
                self.rules = self._load_rules()
                self._rules_mtime = _mtime
        except Exception:
            pass
        attrs = dict(user_attributes)
        if clothing_type:
            attrs["clothing_type"] = clothing_type
        target_category = attrs.get("clothing_type", "").lower()
        gender = attrs.get("gender", "")

        def _item_haystack(item) -> str:
            return " ".join([
                str(item.get("category", "") or ""),
                str(item.get("name", "") or ""),
                str(item.get("description", "") or ""),
                str(item.get("pants_type", "") or ""),
                str(item.get("jeans_type", "") or ""),
            ]).lower().replace("_", " ")

        def _looks_like_shorts(item) -> bool:
            """Detect shorts-like products even when category labels are noisy."""
            pt = (item.get("pants_type", "") or "").lower().strip()
            if pt == "shorts":
                return True
            hay = _item_haystack(item)
            # Match noun-form shorts; avoid excluding short-sleeve tops.
            return (" shorts " in f" {hay} ") or (" short " in f" {hay} " and "short sleeve" not in hay)

        def _matches_clothing_type(item, requested_category: str, item_gender: str) -> bool:
            if not requested_category:
                return True
            terms = self._map_clothing(requested_category, item_gender) or []
            hay = _item_haystack(item)
            pt = (item.get("pants_type", "") or "").lower()
            is_shorts_query = requested_category in {"shorts", "short"}
            is_trousers_query = requested_category in {"trousers", "trouser", "pants"}
            is_top_query = requested_category in {"top", "tops", "t_shirt", "t_shirts", "shirt", "shirts"}

            if is_shorts_query:
                return _looks_like_shorts(item)
            if is_trousers_query:
                return pt != "shorts" and any(t in hay for t in terms)
            if is_top_query:
                return any(t in hay for t in terms) and not _looks_like_shorts(item)
            return any(t in hay for t in terms)

        active_inventory = list(self.inventory)

        # Filter trace: list of {"name", "before", "after"} dicts for each step
        _filter_trace = []

        def _apply_filter(name, fn):
            nonlocal active_inventory
            before = len(active_inventory)
            active_inventory = [it for it in active_inventory if fn(it)]
            after = len(active_inventory)
            _filter_trace.append({"name": name, "before": before, "after": after})

        def is_swim(it):
            kws = ("swim","bikini","swimsuit","beach","cover-up","tankini","boardshort","beachwear")
            hay = " ".join([str(it.get("category","")),str(it.get("name","")),str(it.get("description",""))]).lower()
            return any(k in hay for k in kws)

        if target_category:
            _apply_filter(f"Clothing type ({target_category})", lambda it: _matches_clothing_type(it, target_category, gender))
        _apply_filter("Swimwear exclusion", lambda it: not is_swim(it))

        if gender:
            _apply_filter(f"Gender ({gender})", lambda it: str(it.get("gender","")).lower() == str(gender).lower())

        # ── Sporty occasion: restrict to Nike + Under Armour only ─────────────
        if str(attrs.get("occasion", "")).lower() == "sporty":
            _SPORTY_BRANDS = {"nike", "ua"}
            _apply_filter("Sporty brands (Nike / UA)", lambda it: str(it.get("brand", "")).lower() in _SPORTY_BRANDS)

        # ── Brand filter (user-selected brands only) ──────────────────────────
        _brand_filter = attrs.get("_brand_filter")
        if _brand_filter:
            _bf_lower = {b.lower() for b in _brand_filter}
            _apply_filter(f"Brand ({', '.join(_brand_filter)})", lambda it: str(it.get("brand", "")).lower() in _bf_lower)

        # ── Price range filter ────────────────────────────────────────────────
        _price_range = attrs.get("_price_range")
        if _price_range and len(_price_range) == 2:
            _pmin, _pmax = float(_price_range[0]), float(_price_range[1])
            if _pmin > 0 or _pmax < 500:
                _apply_filter(f"Price (${_pmin:.0f}–${_pmax:.0f})", lambda it: _pmin <= it.get("price", 0) <= _pmax)

        # ── Fit filter ────────────────────────────────────────────────────────
        _fit_filter = attrs.get("_fit_filter")
        if _fit_filter:
            _ff = [f.lower() for f in _fit_filter]
            _apply_filter(f"Fit ({', '.join(_fit_filter)})", lambda it: str(it.get("fit", "")).lower() in _ff)

        query_text, banned_indices = self.generate_query_and_filter(attrs, active_inventory)
        final_inventory = [item for i, item in enumerate(active_inventory) if i not in banned_indices]
        _n_before_rules = len(active_inventory)
        _n_after_rules  = len(final_inventory)
        if _n_after_rules < _n_before_rules:
            _filter_trace.append({"name": "Fashion rules (avoid terms)", "before": _n_before_rules, "after": _n_after_rules})
        if not final_inventory and active_inventory:
            # Rules can be too aggressive for sparse slices like men's shorts.
            # Fall back to the filtered inventory rather than showing an empty state.
            final_inventory = list(active_inventory)
        if not final_inventory:
            return query_text, [], _filter_trace

        inventory_texts = [self._item_text(it) for it in final_inventory]
        inventory_embeddings = self.model.encode(inventory_texts, convert_to_tensor=True)
        query_vec = self.model.encode(query_text, convert_to_tensor=True)
        scores = util.cos_sim(query_vec, inventory_embeddings)[0]

        # ── Build full deduped candidate pool ────────────────────────────────
        # Hard-locked: always 10 ranked + 10 discovery = 20 total for data collection.
        # The display slider in the sidebar controls how many cards are rendered (1-10),
        # but all 20 are saved to profiles.json so feedback on discovery items can be
        # used to tune fashion_rules.json.
        _RANKED_N = 10
        _RANDOM_N = 10
        _POOL_K   = min((_RANKED_N + _RANDOM_N) * 5, len(final_inventory))
        top_results = torch.topk(scores, k=_POOL_K)
        all_candidates = []
        seen = set()
        for score, idx in zip(top_results[0], top_results[1]):
            item = final_inventory[int(idx)]
            dedup_key = (item.get("name","").strip().lower(), item.get("brand","").strip().lower())
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            all_candidates.append({
                "score": float(score),
                "id": item.get("id",""),
                "name": item.get("name",""),
                "gender": item.get("gender",""),
                "brand": item.get("brand",""),
                "description": item.get("description",""),
                "category": item.get("category",""),
                "sleeve_length": item.get("sleeve_length",""),
                "neck_type": item.get("neck_type",""),
                "silhouette": item.get("silhouette",""),
                "color_tone": item.get("color_tone",""),
                "images": item.get("images",[]),
            })

        # ── 50 / 50 split ─────────────────────────────────────────────────────
        ranked_half = all_candidates[:_RANKED_N]
        ranked_ids  = {r["id"] for r in ranked_half}

        # Discovery pool: NO recommendation rules except gender + clothing_type
        # Build a separate discovery_inventory with minimal filters
        discovery_inventory = list(self.inventory)

        # Filter by gender only
        if gender:
            discovery_inventory = [it for it in discovery_inventory if str(it.get("gender","")).lower() == str(gender).lower()]

        # Filter by clothing type only
        if target_category:
            discovery_inventory = [it for it in discovery_inventory if _matches_clothing_type(it, target_category, gender)]
        
        # Remove ranked items from discovery pool (preferred to avoid duplicates).
        # If this empties the pool, fall back to the gender+clothing pool so
        # discovery is always available as requested.
        discovery_base_inventory = list(discovery_inventory)
        discovery_inventory = [it for it in discovery_inventory if it.get("id") not in ranked_ids]
        if not discovery_inventory:
            discovery_inventory = discovery_base_inventory
        
        # Fresh seed each call (passed in from the submit handler via uuid.uuid4())
        # → every session gets a genuinely new random draw for data-collection purposes.
        _rng = random.Random(random_seed)
        
        # Randomly sample 10 from discovery pool (NO scoring, pure random)
        if discovery_inventory:
            random_samples = _rng.sample(discovery_inventory, min(_RANDOM_N, len(discovery_inventory)))
            # Format as candidates with score=0
            random_half = [
                {
                    "score": 0.0,
                    "id": item.get("id",""),
                    "name": item.get("name",""),
                    "gender": item.get("gender",""),
                    "brand": item.get("brand",""),
                    "description": item.get("description",""),
                    "category": item.get("category",""),
                    "sleeve_length": item.get("sleeve_length",""),
                    "neck_type": item.get("neck_type",""),
                    "silhouette": item.get("silhouette",""),
                    "color_tone": item.get("color_tone",""),
                    "images": item.get("images",[]),
                }
                for item in random_samples
            ]
        else:
            random_half = []

        # Tag sources
        for r in ranked_half:
            r["source"] = "ranked"
        for r in random_half:
            r["source"] = "random"

        # Interleave: ranked at even positions (0,2,4…), discovery at odd (1,3,5…)
        results = []
        ri = di = 0
        while ri < len(ranked_half) or di < len(random_half):
            if ri < len(ranked_half):
                results.append(ranked_half[ri]); ri += 1
            if di < len(random_half):
                results.append(random_half[di]); di += 1

        return query_text, results, _filter_trace


# ── Why-This explanation helpers ─────────────────────────────────────────────
_JUNK = {"", "unknown", "none", "n/a", "nan", "null", "na"}

def _val(v):
    s = (v or "").strip().lower()
    return (v or "").strip() if s not in _JUNK else ""

_OCCASION_DESC = {
    "sporty":      "active and breathable — built for movement",
    "formals":     "structured and polished — sharp enough for formal settings",
    "event":       "elevated and styled — stands out for occasions",
    "casual":      "relaxed and comfortable — easy to wear all day",
}
_FACE_NECK = {
    ("oval",   "v-neck"):  "V-necks elongate an oval face beautifully",
    ("oval",   "crew"):    "Crew necks sit neatly on a naturally balanced oval face",
    ("oval",   "scoop"):   "Scoop necks open up the neckline and complement oval faces",
    ("round",  "v-neck"):  "V-necks visually lengthen and slim a round face",
    ("round",  "scoop"):   "Scoop necklines add width, balancing a round face",
    ("square", "round"):   "Rounded necklines soften the angles of a square jaw",
    ("square", "scoop"):   "Scoop necks introduce curves against a defined square jawline",
    ("square", "cowl"):    "Draped cowl necks add soft movement against a square jaw",
    ("heart",  "boat"):    "Boat necks widen the shoulder line, balancing a narrower chin",
    ("heart",  "scoop"):   "Scoop necklines draw the eye down from a wider forehead",
    ("oblong", "boat"):    "Boat necklines add horizontal breadth to a long face",
    ("oblong", "crew"):    "Crew necks break the vertical line of a longer face shape",
    ("oblong", "square"):  "Square necklines interrupt the vertical and broaden a long face",
}
_TONE_COLOR = {
    ("warm", "black"):    "Black creates bold contrast with warm undertones",
    ("warm", "white"):    "Crisp white pops bright against warm skin",
    ("warm", "beige"):    "Beige blends harmoniously with warm undertones",
    ("warm", "brown"):    "Earthy brown is a natural partner to warm skin tones",
    ("warm", "khaki"):    "Khaki reads as a warm neutral — a natural fit",
    ("warm", "navy"):     "Navy gives a rich, flattering contrast against warm skin",
    ("warm", "red"):      "Red amplifies the warmth in your skin tone",
    ("warm", "olive"):    "Olive green sits in the same warm colour family as your undertone",
    ("cool", "navy"):     "Navy aligns naturally with cool undertones",
    ("cool", "gray"):     "Gray echoes cool undertones for a polished, cohesive look",
    ("cool", "white"):    "White is classically clean against cool skin",
    ("cool", "black"):    "Black gives sharp, elegant contrast on cool undertones",
    ("cool", "blue"):     "Blue complements cool undertones — a natural match",
    ("cool", "purple"):   "Purple shares the cool spectrum and flatters cool skin",
    ("cool", "lavender"): "Lavender is a soft cool-toned choice that harmonises well",
}
_BODY_SIL = {
    ("hourglass",         "fitted"): "A fitted silhouette highlights the natural waist of an hourglass figure",
    ("hourglass",         "wrap"):   "Wrap styles celebrate the hourglass waist",
    ("pear",              "a-line"): "A-line flares gently from the waist, balancing a pear shape",
    ("pear",              "flare"):  "Flared hems draw the eye outward at the hip for balance",
    ("rectangle",         "wrap"):   "Wrap styles define a waist on a straighter rectangle frame",
    ("rectangle",         "peplum"): "Peplum adds a flare at the hip, creating curves on a rectangle figure",
    ("inverted_triangle", "a-line"): "A-line fullness at the hem balances broader shoulders",
}
_HAIR_COLOR_PAIR = {
    ("blonde",   "navy"):  "Navy creates a cool contrasting backdrop for blonde hair",
    ("brunette", "brown"): "Brown tones echo the warmth of brunette hair naturally",
    ("black",    "white"): "The black–white contrast pops against dark hair",
    ("black",    "red"):   "Bold red reads vividly against dark hair",
    ("red",      "green"): "Green is a complementary colour to red hair on the wheel",
    ("blonde",   "white"): "Light-on-light creates a soft airy look for blonde hair",
}


def build_explanation(item, user_attrs, query_text):
    """Build a crisp, item-specific explanation as bullet points."""
    bullets = []
    brand   = _val(item.get("brand"))
    cat     = _val(item.get("category"))
    sleeve  = _val(item.get("sleeve_length"))
    neck    = _val(item.get("neck_type"))
    sil     = _val(item.get("silhouette"))
    color_t = _val(item.get("color_tone"))
    desc    = (item.get("description") or "").strip()
    score   = item.get("adjusted_score", item.get("score", 0))

    occ      = _val(user_attrs.get("occasion"))
    face     = _val(user_attrs.get("face_shape"))
    body     = _val(user_attrs.get("body_type"))
    tone     = _val(user_attrs.get("skin_tone"))
    hair_cat = _val(user_attrs.get("hair_color"))

    label     = f"{brand} " if brand else ""
    cat_title = cat.title() if cat else "This piece"

    # 1. Occasion — garment-specific
    if occ and cat:
        occ_desc = _OCCASION_DESC.get(occ.lower(), f"suited for {occ}")
        bullets.append(f"**{label}{cat_title}** — {occ_desc}")
    elif cat:
        bullets.append(f"**{label}{cat_title}**")

    # 2. Neckline × face shape (skipped if neckline is unknown)
    if face and neck:
        key  = (face.lower(), neck.lower())
        text = _FACE_NECK.get(key, f"{neck.title()} neckline is flattering on a {face} face shape")
        bullets.append(text)

    # 3. Silhouette × body type
    if body and sil:
        key  = (body.lower(), sil.lower())
        text = _BODY_SIL.get(key, f"The {sil} cut works well for a {body.replace('_', ' ')} shape")
        bullets.append(text)

    # 4. Color × skin tone (specific pairing)
    if tone and color_t:
        key  = (tone.lower(), color_t.lower())
        text = _TONE_COLOR.get(key, f"{color_t.title()} is a good pick for {tone} undertones")
        bullets.append(text)
    elif color_t and not tone:
        bullets.append(f"Available in {color_t} — a versatile everyday choice")

    # 5. Hair colour bonus pairing
    if hair_cat and color_t:
        hkey = (hair_cat.lower(), color_t.lower())
        if hkey in _HAIR_COLOR_PAIR:
            bullets.append(_HAIR_COLOR_PAIR[hkey])

    # 6. Sleeve callout if not redundant
    if sleeve and sleeve.lower() not in (cat or "").lower() and len(bullets) < 4:
        bullets.append(f"{sleeve.title()} sleeves — adds to the look and everyday wearability")

    # 7. Fallback — pull from product description
    if len(bullets) <= 1 and desc:
        snippet = desc[:110]
        if len(desc) > 110:
            snippet = snippet.rsplit(" ", 1)[0] + "…"
        bullets.append(f'"{snippet}"')

    if not bullets:
        rating = "strong" if score > 0.65 else ("good" if score > 0.45 else "moderate")
        return f"A {rating} match based on your combined style profile."

    return "\n\n".join(f"• {b}" for b in bullets[:4])


def apply_feedback_rerank(results, like_boost=0.2, dislike_penalty=0.2):
    """Rerank ONLY ranked items based on like/dislike feedback.
    
    Discovery items (source='random') stay static and unaffected by feedback,
    preserving their exploratory purpose.
    """
    reranked = []
    ranked_items = []
    discovery_items = []
    
    # Separate ranked and discovery
    for item in results:
        if item.get("source") == "random":
            discovery_items.append(item)
        else:
            ranked_items.append(item)
    
    # Rerank ONLY ranked items based on feedback
    for item in ranked_items:
        base = item["score"]
        pid  = item.get("id")
        fb   = st.session_state.feedback.get(pid)
        adj  = base + (like_boost if fb == "like" else -dislike_penalty if fb == "dislike" else 0)
        new_item = dict(item)
        new_item["adjusted_score"] = adj
        reranked.append(new_item)
    
    # Sort ranked items by adjusted score
    reranked.sort(key=lambda x: x["adjusted_score"], reverse=True)
    
    # Append discovery items unchanged (no feedback reranking, static/exploratory)
    reranked.extend(discovery_items)
    
    return reranked


@st.cache_resource
def _load_recommender(_csv_mtime: float = 0.0, _code_mtime: float = 0.0):
    """Cache-key args force rebuild when data or app code changes."""
    return ConfigurableRecommender(PRODUCTS_CSV, RULES_FILE, IMAGES_ROOT)


def _recommender():
    """Call _load_recommender with the current products.csv mtime so the cache
    auto-invalidates whenever the file is updated without restarting Streamlit."""
    try:
        mtime = os.path.getmtime(PRODUCTS_CSV)
    except OSError:
        mtime = 0.0
    try:
        code_mtime = os.path.getmtime(__file__)
    except OSError:
        code_mtime = 0.0
    return _load_recommender(mtime, code_mtime)


# ═══════════════════════════════════════════════════════════════════════════════
# ATTRIBUTE BRIDGE  — Task 1 attrs → Task 2 attrs
# ═══════════════════════════════════════════════════════════════════════════════

def _body_type_to_t2(body_type_t1: str) -> str:
    """Convert Task 1 title-case body type to Task 2 snake_case."""
    mapping = {
        "Hourglass": "hourglass",
        "Inverted Triangle": "inverted_triangle",
        "Pear": "pear",
        "Rectangle": "rectangle",
    }
    return mapping.get(body_type_t1, (body_type_t1 or "").lower())


def _undertone_to_skin_tone(undertone: str) -> str:
    """Map Task 1 undertone to Task 2 skin_tone options."""
    mapping = {"Warm": "warm", "Cool": "cool", "Neutral": "warm"}
    return mapping.get(undertone, "")


def _neck_length_to_t2(neck_length: str) -> str:
    """Map Task 1 neck_length to Task 2 options (short/average/long)."""
    mapping = {"Short": "short", "Medium": "average", "Long": "long"}
    return mapping.get(neck_length, "")


def _hex_to_hair_category(hex_color: str) -> str:
    """Classify a hair hex color into a Task 2 category."""
    if not hex_color or hex_color == "#000000":
        return ""
    try:
        r, g, b = hex_to_rgb(hex_color)
        hsv = cv2.cvtColor(np.array([[[r, g, b]]], dtype=np.uint8), cv2.COLOR_RGB2HSV)[0][0]
        h_val, s_val, v_val = int(hsv[0]), int(hsv[1]), int(hsv[2])
        if v_val < 50:
            return "black"
        if s_val < 40 and v_val > 160:
            return "gray"
        # Red/auburn hues
        if (h_val < 15 or h_val > 165) and s_val > 60:
            return "red"
        # Blonde: high brightness, low saturation or yellowish
        if v_val > 160 and s_val < 120:
            return "blonde"
        return "brunette"
    except Exception:
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "step":             0,
        "user_id":          None,
        "user_name":        None,
        "photo_captures":   {},
        "view_results":     {},
        "body_type":        None,
        "final_attrs":      None,
        "returning_attrs":  None,
        "last_session_at":  None,
        # recommender state
        "feedback":         {},
        "rec_results":      None,
        "rec_query":        "",
        "rec_filter_trace": [],
        "image_index":      {},
        "tryon_selected":   [],
        "outfit_components": {"top": None, "bottom": None, "shoes": None},
        "outfit_locks":      {"top": False, "bottom": False, "shoes": False},
        # try-on state
        "tryon_count":            100,   # generation budget
        "tryon_item":           None,  # product dict selected for try-on
        "tryon_image":          None,  # local path of shown garment image
        "tryon_result":         None,  # result bytes or error string
        "tryon_expected_item_id": None,  # item id expected at submit time
        "tryon_expected_image":   None,  # image path expected at submit time
        "tryon_expected_component": None, # top/bottom/shoes expected at submit time
        "tryon_person_override": None,  # override photo bytes for try-on (not used for analysis)
        "_selected_garment_type": "full",  # selected garment type for segmentation
        "tryon_mode": "single",            # "single" or "combo"
        "tryon_combo_bottom_image": None,  # bottom garment path for combo try-on
        "tryon_generate_requested": False, # legacy latch across reruns
        "tryon_generate_request": None,    # persistent generate request payload
        "tryon_request_inflight": False,   # prevents duplicate API calls per click
        "tryon_last_request_sig": None,    # de-duplicate accidental repeated submits
        "tryon_selection_request": None,   # pending card -> try-on selection handoff
        "current_experiment_id": None,  # track which experiment session we're in
        "new_profile":          False, # True immediately after account creation
        "current_session_id":   None,  # analysis session_id written to DB
        "current_rec_session_id": None, # rec session_id written to DB
        "style_quiz":   None,  # None=not taken; {}=skipped; {…}=filled
        "theme":        "dark",   # "dark" | "light" | "comfort"
        "fullscreen":   False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _infer_component(item: dict) -> str | None:
    """Infer coarse outfit component from item category/name/description."""
    cat = str(item.get("category", "")).lower()
    name = str(item.get("name", "")).lower()
    desc = str(item.get("description", "")).lower()
    text = " ".join([cat, name, desc])

    shoe_terms = ("shoe", "sneaker", "loafer", "heel", "boot", "sandal", "mule", "clog", "flat")
    bottom_terms = ("pant", "trouser", "jean", "short", "skirt", "legging", "jogger")
    top_terms = ("top", "shirt", "tee", "t-shirt", "blouse", "hoodie", "sweatshirt", "polo", "tank")

    if any(t in text for t in shoe_terms):
        return "shoes"
    if any(t in text for t in bottom_terms):
        return "bottom"
    if any(t in text for t in top_terms):
        return "top"
    return None


def _validate_tryon_selection(item: dict, gpath: str) -> tuple[bool, str]:
    """Ensure current try-on submit still points to the selected item/image."""
    if not item or not gpath:
        return False, "No try-on item/image selected."

    images = item.get("images", []) or []
    if images:
        def _norm(p: str) -> str:
            return os.path.normcase(os.path.normpath(str(p or "")))
        gpath_norm = _norm(gpath)
        image_norms = {_norm(p) for p in images}
        # Allow equivalent normalized paths and basename-equivalent paths.
        if gpath_norm not in image_norms:
            gbase = os.path.basename(gpath_norm)
            ibases = {os.path.basename(p) for p in image_norms}
            if gbase not in ibases:
                return False, "Selected garment image is out of sync with this product. Please reselect the item."

    expected_id = st.session_state.get("tryon_expected_item_id")
    expected_img = st.session_state.get("tryon_expected_image")
    if expected_id and str(item.get("id", "")) != str(expected_id):
        return False, "Try-on item changed unexpectedly. Please choose the garment again."
    if expected_img:
        g_norm = os.path.normcase(os.path.normpath(str(gpath)))
        e_norm = os.path.normcase(os.path.normpath(str(expected_img)))
        if g_norm != e_norm and os.path.basename(g_norm) != os.path.basename(e_norm):
            return False, "Try-on garment image changed unexpectedly. Please choose the garment again."

    return True, ""


# ═══════════════════════════════════════════════════════════════════════════════
# STEP CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

STEPS = [
    "Identity",
    "Capture",
    "Body Type",
    "Analysis",
    "Results",
    "Recommendations",
    "Try-On",
]
HRVITON_ROOT = Path(os.getenv("HRVITON_ROOT", str(ROOT_DIR / "HR-VITON-main"))).resolve()
HRVITON_PYTHON = os.getenv("HRVITON_PYTHON", "").strip()  # optional dedicated env python
HRVITON_GPU_IDS = os.getenv("HRVITON_GPU_IDS", "0").strip()
HRVITON_FORCE_CPU = os.getenv("HRVITON_FORCE_CPU", "0").strip().lower() in {"1", "true", "yes"}
HRVITON_TOCG_CKPT = os.getenv("HRVITON_TOCG_CKPT", str(HRVITON_ROOT / "eval_models" / "weights" / "v0.1" / "mtviton.pth")).strip()
HRVITON_GEN_CKPT = os.getenv("HRVITON_GEN_CKPT", str(HRVITON_ROOT / "eval_models" / "weights" / "v0.1" / "gen.pth")).strip()
HRVITON_RUNTIME_ROOT = Path(os.getenv("HRVITON_RUNTIME_ROOT", str(ROOT_DIR / "Tryons" / "hrviton_runtime"))).resolve()

# ── DCI-VTON constants ────────────────────────────────────────────────────────
DCIVTON_ROOT   = Path(os.getenv("DCIVTON_ROOT",   str(ROOT_DIR / "DCI-VTON-Virtual-Try-On-main"))).resolve()
DCIVTON_CKPT   = os.getenv("DCIVTON_CKPT",   str(ROOT_DIR / "DCI models" / "viton512_v2.ckpt")).strip()
DCIVTON_PYTHON = os.getenv("DCIVTON_PYTHON", sys.executable).strip()
DCIVTON_STEPS  = int(os.getenv("DCIVTON_STEPS", "10"))
DCIVTON_INFER  = str(DCIVTON_ROOT / "dcivton_infer.py")
_DCIVTON_RUNTIME_MODULE = None
_DCIVTON_MODEL = None
_DCIVTON_DEVICE = None

# ── IDM-VTON constants ────────────────────────────────────────────────────────
IDMVTON_ROOT = Path(os.getenv("IDMVTON_ROOT", str(ROOT_DIR / "IDM-VTON-main"))).resolve()
# Use isolated .venv_idm in IDM-VTON-main folder; fallback to main env if not found
_IDMVTON_VENV = IDMVTON_ROOT / ".venv_idm" / "Scripts" / "python.exe"
IDMVTON_PYTHON = os.getenv("IDMVTON_PYTHON", str(_IDMVTON_VENV) if _IDMVTON_VENV.exists() else sys.executable).strip()
IDMVTON_STEPS = int(os.getenv("IDMVTON_STEPS", "30"))
IDMVTON_WIDTH = int(os.getenv("IDMVTON_WIDTH", "768"))
IDMVTON_HEIGHT = int(os.getenv("IDMVTON_HEIGHT", "1024"))
IDMVTON_TIMEOUT = int(os.getenv("IDMVTON_TIMEOUT", "1800"))
IDMVTON_RUNTIME_ROOT = Path(os.getenv("IDMVTON_RUNTIME_ROOT", str(ROOT_DIR / "Tryons" / "idmvton_runtime"))).resolve()
IDMVTON_PREVIEW_STEPS = int(os.getenv("IDMVTON_PREVIEW_STEPS", "12"))
IDMVTON_PREVIEW_WIDTH = int(os.getenv("IDMVTON_PREVIEW_WIDTH", "576"))
IDMVTON_PREVIEW_HEIGHT = int(os.getenv("IDMVTON_PREVIEW_HEIGHT", "768"))
IDMVTON_HQ_STEPS = int(os.getenv("IDMVTON_HQ_STEPS", str(IDMVTON_STEPS)))
IDMVTON_HQ_WIDTH = int(os.getenv("IDMVTON_HQ_WIDTH", str(IDMVTON_WIDTH)))
IDMVTON_HQ_HEIGHT = int(os.getenv("IDMVTON_HQ_HEIGHT", str(IDMVTON_HEIGHT)))
IDMVTON_USE_WORKER = os.getenv("IDMVTON_USE_WORKER", "1").strip().lower() in {"1", "true", "yes", "on"}
IDMVTON_WORKER_SCRIPT = Path(os.getenv("IDMVTON_WORKER_SCRIPT", str(IDMVTON_ROOT / "fits_idm_worker.py"))).resolve()
# Default inference steps (optimized from 30 to 20 for local RTX 2000 Ada)
IDMVTON_STEPS_PREVIEW = int(os.getenv("IDMVTON_STEPS_PREVIEW", "8"))   # Quick iteration
IDMVTON_STEPS_HQ = int(os.getenv("IDMVTON_STEPS_HQ", "20"))           # Quality (was 30, optimized)
IDMVTON_STEPS = int(os.getenv("IDMVTON_STEPS", str(IDMVTON_STEPS_HQ)))  # Backward compatible
IDMVTON_WIDTH = int(os.getenv("IDMVTON_WIDTH", "768"))
IDMVTON_HEIGHT = int(os.getenv("IDMVTON_HEIGHT", "1024"))
IDMVTON_TIMEOUT = int(os.getenv("IDMVTON_TIMEOUT", "1800"))
# Command template placeholders: {python} {root} {person} {cloth} {output} {steps} {width} {height}
IDMVTON_COMMAND = os.getenv(
    "IDMVTON_COMMAND",
    "{python} fits_single_infer.py --person {person} --cloth {cloth} --output {output} --steps {steps} --width {width} --height {height}",
).strip()

TRYON_LOCAL_MODE = os.getenv("TRYON_LOCAL_MODE", "idmvton").strip().lower()  # simple|dcivton|idmvton|hrviton


def _safe_get_secret(key: str, default: str = "") -> str:
    """Read Streamlit secret without raising when secrets.toml is missing."""
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

BODY_TYPES = ["Hourglass", "Inverted Triangle", "Pear", "Rectangle"]
VIEWS = [
    ("front", "Full Body — stand ~1.5 m back, full body visible"),
    ("face",  "Face — close-up, chin to forehead centred"),
]
# Countdown timer (seconds) per view
VIEW_TIMER = {"face": 3, "front": 10}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

# ── Global CSS ────────────────────────────────────────────────────────────────
_FITS_GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ══════════════════════════════════════════════════════════════════
   DARK MODE — Premium Fashion App
   Palette:
     bg-base:    #06070d   deep space black
     bg-raised:  #0c0e18   slightly lifted
     bg-card:    #101420   card surfaces
     bg-glass:   rgba(16,20,32,0.7) glass
     bg-input:   #090b14   inputs
     border:     #1a2236   quiet borders
     border-hi:  #2a3a58   hover borders
     text-pri:   #eef1f7   near white
     text-sec:   #7a8fa8   muted
     text-dim:   #3a4f65   very dim
     accent:     #5ec8c8   teal
     accent-hi:  #7de8e8   bright teal
     accent-dk:  #3ea8a8   dark teal
     accent-glow: rgba(94,200,200,0.2)
   ══════════════════════════════════════════════════════════════════ */

:root {
    --bg:          #06070d;
    --bg-raised:   #0c0e18;
    --bg-card:     #101420;
    --bg-input:    #090b14;
    --border:      #1a2236;
    --border-hi:   #2a3a58;
    --text-pri:    #f8faff;
    --text-sec:    #b0c8de;
    --text-dim:    #5a7a96;
    --accent:      #2fe5e5;
    --accent-hi:   #8afefe;
    --accent-dk:   #11b8c8;
    --accent-glow: rgba(47,229,229,0.28);
}

/* ── Global reset & font ──────────────────────────────────────────── */
html, body, [class*="css"],
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > div,
[data-testid="stAppViewBlockContainer"],
[data-testid="stVerticalBlock"],
[data-testid="stHeader"],
.main, .main > div,
.main .block-container,
.block-container,
[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"],
[data-testid="stToolbar"] {
    font-family: 'Inter', 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-pri) !important;
    background-color: transparent !important;
}
.stApp,
[data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
}
[data-testid="stHeader"] {
    background: rgba(6,7,13,0.8) !important;
    backdrop-filter: blur(20px) saturate(1.95) !important;
    -webkit-backdrop-filter: blur(20px) saturate(1.95) !important;
    border-bottom: 1px solid rgba(26,34,54,0.8) !important;
    box-shadow: 0 1px 0 rgba(47,229,229,0.09) !important;
}

/* ── Scrollbar ────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, var(--border-hi), var(--border));
    border-radius: 99px;
}
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ══════════════════════════════════════════════════════════════════
   SIDEBAR
   ══════════════════════════════════════════════════════════════════ */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, #080a14 0%, #060810 100%) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: 4px 0 32px rgba(0,0,0,0.5) !important;
}
section[data-testid="stSidebar"] * {
    color: var(--text-sec) !important;
    background-color: transparent !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: var(--text-pri) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.02em !important;
}
section[data-testid="stSidebar"] label {
    color: var(--text-dim) !important;
    font-size: 0.68rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 700 !important;
}
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"],
section[data-testid="stSidebar"] .stSelectbox > div > div,
section[data-testid="stSidebar"] .stSlider > div {
    background: var(--bg-input) !important;
    border-color: var(--border) !important;
}
section[data-testid="stSidebar"] .stButton > button {
    background: rgba(26,34,54,0.6) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-sec) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    backdrop-filter: blur(8px) !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: rgba(47,229,229,0.09) !important;
    box-shadow: 0 0 18px rgba(47,229,229,0.22) !important;
}
section[data-testid="stSidebar"] hr {
    border-color: var(--border) !important;
    opacity: 0.5 !important;
}
section[data-testid="stSidebar"] .stFormSubmitButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dk) 100%) !important;
    border: none !important;
    color: #04060c !important;
    font-weight: 800 !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 24px rgba(47,229,229,0.36) !important;
    letter-spacing: 0.02em !important;
}
section[data-testid="stSidebar"] .stFormSubmitButton > button p,
section[data-testid="stSidebar"] .stFormSubmitButton > button span,
section[data-testid="stSidebar"] .stFormSubmitButton > button div {
    color: #04060c !important;
}

/* ══════════════════════════════════════════════════════════════════
   MAIN AREA — typography
   ══════════════════════════════════════════════════════════════════ */
.main h1, .main h2, .main h3,
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3 {
    color: var(--text-pri) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.025em !important;
}
.main p, .main span, .main li, .main label,
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] label {
    color: var(--text-sec) !important;
}
.main a { color: var(--accent) !important; text-decoration: none !important; }

/* ══════════════════════════════════════════════════════════════════
   INPUTS
   ══════════════════════════════════════════════════════════════════ */
.main input,
.main textarea,
.main [data-baseweb="select"],
.main [data-baseweb="input"] input,
.main [data-testid="stTextInput"] input,
.main .stTextInput input,
.main .stTextArea textarea,
.main .stNumberInput input,
.main .stSelectbox [data-baseweb="select"],
[data-testid="stAppViewContainer"] input,
[data-testid="stAppViewContainer"] textarea,
[data-testid="stAppViewContainer"] [data-baseweb="select"] {
    background-color: var(--bg-input) !important;
    color: var(--text-pri) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 12px !important;
    font-size: 0.92rem !important;
    caret-color: var(--accent) !important;
    transition: border-color 0.18s ease, box-shadow 0.18s ease !important;
}
.main input:focus,
.main textarea:focus,
[data-testid="stAppViewContainer"] input:focus,
[data-testid="stAppViewContainer"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(47,229,229,0.2), 0 0 20px rgba(47,229,229,0.12) !important;
    outline: none !important;
}
.main input::placeholder,
.main textarea::placeholder {
    color: var(--text-dim) !important;
}
.main [data-testid="stTextInput"] label,
.main .stTextInput label,
.main .stTextArea label,
.main .stNumberInput label,
.main .stSelectbox label,
[data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] {
    color: var(--text-sec) !important;
    font-weight: 700 !important;
    font-size: 0.76rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
/* Label inner spans must also be bright */
.main [data-testid="stWidgetLabel"] p,
.main [data-testid="stWidgetLabel"] span,
[data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] p,
[data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] span {
    color: var(--text-sec) !important;
}

/* Global saturation lift for dim displays; per-theme overrides adjust this. */
.stApp,
[data-testid="stAppViewContainer"],
section[data-testid="stSidebar"] {
    filter: saturate(1.14) !important;
}

[data-baseweb="popover"],
[data-baseweb="menu"],
[data-baseweb="popover"] ul,
[data-baseweb="menu"] ul {
    background-color: #0f1320 !important;
    border: 1px solid var(--border-hi) !important;
    box-shadow: 0 16px 48px rgba(0,0,0,0.6) !important;
    border-radius: 12px !important;
}
[data-baseweb="popover"] li,
[data-baseweb="menu"] li {
    background-color: transparent !important;
    color: var(--text-sec) !important;
    border-radius: 8px !important;
    margin: 2px 4px !important;
    font-size: 0.88rem !important;
}
[data-baseweb="menu"] li:hover,
[data-baseweb="popover"] li:hover {
    background: rgba(47,229,229,0.11) !important;
    color: var(--text-pri) !important;
}

/* ── Form wrapper ─────────────────────────────────────────────────── */
.main [data-testid="stForm"],
[data-testid="stAppViewContainer"] [data-testid="stForm"] {
    background: rgba(12,14,24,0.7) !important;
    border: 1px solid var(--border) !important;
    border-radius: 20px !important;
    padding: 1.8rem !important;
    backdrop-filter: blur(20px) !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.03) !important;
}

/* ── Radio / checkbox ─────────────────────────────────────────────── */
.main [data-testid="stRadio"] > div,
.main [data-testid="stCheckbox"] > div {
    background-color: transparent !important;
}
.main [data-testid="stRadio"] label,
.main [data-testid="stCheckbox"] label {
    color: var(--text-sec) !important;
}

/* ══════════════════════════════════════════════════════════════════
   BUTTONS
   ══════════════════════════════════════════════════════════════════ */
/* ── Regular buttons ──────────────────────────────────────────────── */
.main .stButton > button,
[data-testid="stAppViewContainer"] .stButton > button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    border: 1px solid var(--border-hi) !important;
    background: rgba(16,20,32,0.8) !important;
    color: var(--text-sec) !important;
    transition: all 0.2s cubic-bezier(0.4,0,0.2,1) !important;
    backdrop-filter: blur(8px) !important;
    letter-spacing: 0.01em !important;
}
/* Force button inner text elements to inherit button color (beat global span/p rule) */
.main .stButton > button p,
.main .stButton > button span,
.main .stButton > button div,
[data-testid="stAppViewContainer"] .stButton > button p,
[data-testid="stAppViewContainer"] .stButton > button span,
[data-testid="stAppViewContainer"] .stButton > button div {
    color: inherit !important;
}
.main .stButton > button:hover,
[data-testid="stAppViewContainer"] .stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent-hi) !important;
    background: rgba(47,229,229,0.09) !important;
    box-shadow: 0 0 0 1px rgba(47,229,229,0.28), 0 4px 22px rgba(47,229,229,0.17) !important;
    transform: translateY(-1px) !important;
}

/* ── Primary / submit buttons ─────────────────────────────────────── */
.main div[data-testid="stButton"] button[kind="primary"],
.main .stFormSubmitButton > button,
[data-testid="stAppViewContainer"] .stFormSubmitButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dk) 100%) !important;
    border: none !important;
    color: #03050a !important;
    font-weight: 800 !important;
    letter-spacing: 0.03em !important;
    box-shadow: 0 4px 24px rgba(47,229,229,0.42), 0 1px 0 rgba(255,255,255,0.12) inset !important;
}
/* Force inner elements of primary/submit buttons to use dark text */
.main div[data-testid="stButton"] button[kind="primary"] p,
.main div[data-testid="stButton"] button[kind="primary"] span,
.main div[data-testid="stButton"] button[kind="primary"] div,
.main .stFormSubmitButton > button p,
.main .stFormSubmitButton > button span,
.main .stFormSubmitButton > button div,
[data-testid="stAppViewContainer"] .stFormSubmitButton > button p,
[data-testid="stAppViewContainer"] .stFormSubmitButton > button span,
[data-testid="stAppViewContainer"] .stFormSubmitButton > button div {
    color: #03050a !important;
}
.main div[data-testid="stButton"] button[kind="primary"]:hover,
.main .stFormSubmitButton > button:hover,
[data-testid="stAppViewContainer"] .stFormSubmitButton > button:hover {
    background: linear-gradient(135deg, var(--accent-hi) 0%, var(--accent) 100%) !important;
    box-shadow: 0 6px 32px rgba(47,229,229,0.52) !important;
    transform: translateY(-2px) !important;
}

/* ── Download button ──────────────────────────────────────────────── */
.main .stDownloadButton > button {
    background: rgba(16,20,32,0.8) !important;
    border: 1px solid var(--border-hi) !important;
    color: var(--text-sec) !important;
    border-radius: 12px !important;
}

/* ── Alerts ────────────────────────────────────────────────────────── */
.main [data-testid="stAlert"],
.main div[data-baseweb="notification"] {
    border-radius: 14px !important;
    border: 1px solid var(--border-hi) !important;
    background: rgba(12,14,24,0.85) !important;
    backdrop-filter: blur(12px) !important;
}

/* ── Progress bar ─────────────────────────────────────────────────── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--accent-dk), var(--accent), var(--accent-hi)) !important;
    background-size: 200% 100% !important;
    border-radius: 99px !important;
    animation: gradientShift 2s linear infinite !important;
}
.stProgress > div > div {
    background: var(--border) !important;
    border-radius: 99px !important;
}

/* ── Expanders ────────────────────────────────────────────────────── */
.main details,
[data-testid="stExpander"] {
    background: rgba(12,14,24,0.5) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    backdrop-filter: blur(8px) !important;
    transition: border-color 0.2s ease !important;
}
.main details:hover,
[data-testid="stExpander"]:hover {
    border-color: var(--border-hi) !important;
}
.main details summary span { color: var(--text-sec) !important; font-weight: 600 !important; }

/* ── Captions ─────────────────────────────────────────────────────── */
.main [data-testid="stCaptionContainer"] span { color: var(--text-dim) !important; }

/* ── Dividers ─────────────────────────────────────────────────────── */
.main hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    opacity: 0.6 !important;
}

/* ── Columns & containers ─────────────────────────────────────────── */
.main [data-testid="column"],
.main [data-testid="stVerticalBlock"],
.main [data-testid="stHorizontalBlock"] {
    background-color: transparent !important;
}

/* ── Images ───────────────────────────────────────────────────────── */
.main [data-testid="stImage"] {
    border-radius: 12px;
    overflow: hidden;
}

/* ── Metrics ──────────────────────────────────────────────────────── */
[data-testid="stMetricValue"] {
    color: var(--text-pri) !important;
    font-weight: 800 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
[data-testid="stMetricLabel"] { color: var(--text-dim) !important; }

/* ── Tabs ─────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(12,14,24,0.5) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    border: 1px solid var(--border) !important;
    gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
    color: var(--text-dim) !important;
    font-weight: 600 !important;
    background: transparent !important;
    border-radius: 8px !important;
    transition: all 0.18s ease !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    background: rgba(94,200,200,0.08) !important;
    border-bottom-color: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] { background: transparent !important; }

/* ── Spinner ──────────────────────────────────────────────────────── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Markdown containers ──────────────────────────────────────────── */
.main [data-testid="stMarkdownContainer"],
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] {
    background-color: transparent !important;
}

/* ── Camera input ──────────────────────────────────────────────────── */
.main [data-testid="stCameraInput"] video,
.main [data-testid="stCameraInput"] img {
    border-radius: 14px !important;
    border: 1px solid var(--border-hi) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4) !important;
}
.main [data-testid="stCameraInput"] button {
    background: linear-gradient(135deg, var(--accent), var(--accent-dk)) !important;
    border: none !important;
    color: #04060c !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
}

/* ── Toast / snackbar ─────────────────────────────────────────────── */
[data-testid="stToast"] {
    background: rgba(12,14,24,0.95) !important;
    color: var(--text-pri) !important;
    border: 1px solid var(--border-hi) !important;
    border-radius: 14px !important;
    backdrop-filter: blur(20px) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5) !important;
}

/* ══════════════════════════════════════════════════════════════════
   PRODUCT CARDS
   ══════════════════════════════════════════════════════════════════ */
.product-card {
    border-radius: 20px;
    background: var(--bg-card) !important;
    border: 1px solid var(--border);
    box-shadow: 0 4px 16px rgba(0,0,0,0.4), 0 1px 0 rgba(255,255,255,0.02) inset;
    transition: transform 0.25s cubic-bezier(0.4,0,0.2,1),
                box-shadow 0.25s cubic-bezier(0.4,0,0.2,1),
                border-color 0.25s ease;
    overflow: hidden;
    margin-bottom: 1rem;
    position: relative;
}
.product-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(94,200,200,0.15), transparent);
    pointer-events: none;
}
.card-img-wrap {
    width: 100%;
    height: 320px;
    overflow: hidden;
    border-radius: 0;
    background: #09091a;
}
.card-img-wrap img,
.product-card [data-testid="stImage"] img {
    width: 100% !important;
    height: 320px !important;
    object-fit: cover !important;
    object-position: top center !important;
    display: block !important;
    transition: transform 0.4s ease !important;
}
.product-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 24px 60px rgba(0,0,0,0.6), 0 0 0 1px rgba(94,200,200,0.15);
    border-color: rgba(94,200,200,0.25);
}
.product-card:hover .card-img-wrap img,
.product-card:hover [data-testid="stImage"] img {
    transform: scale(1.03) !important;
}
.card-body { padding: 1rem 1.1rem 1.2rem; }
.product-brand {
    font-size: 0.65rem; font-weight: 800; letter-spacing: 0.18em;
    text-transform: uppercase; color: var(--accent) !important; margin-bottom: 4px;
    opacity: 0.9;
}
.product-title {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700; font-size: 0.97rem; color: var(--text-pri) !important;
    line-height: 1.3; margin: 0 0 0.5rem;
    letter-spacing: -0.01em;
}
.score-pill {
    display: inline-flex; align-items: center; gap: 4px;
    background: rgba(94,200,200,0.08) !important;
    color: var(--accent) !important;
    border-radius: 999px; padding: 3px 12px;
    font-size: 0.7rem; font-weight: 700;
    border: 1px solid rgba(94,200,200,0.2);
    letter-spacing: 0.02em;
}
.badge-sporty {
    display: inline-block;
    background: rgba(251,113,133,0.1) !important;
    color: #fb7185 !important;
    border-radius: 999px; padding: 2px 10px;
    font-size: 0.64rem; font-weight: 800;
    border: 1px solid rgba(251,113,133,0.22); margin-left: 5px;
    letter-spacing: 0.04em;
}
.filter-pill {
    display: inline-block;
    background: rgba(99,179,237,0.08) !important;
    color: #63b3ed !important;
    border-radius: 999px; padding: 3px 12px;
    font-size: 0.7rem; font-weight: 600;
    margin: 2px 3px;
    border: 1px solid rgba(99,179,237,0.2);
}

/* ── Rec section hero ─────────────────────────────────────────────── */
.rec-hero {
    background: linear-gradient(135deg, var(--bg-card) 0%, #0e1828 60%, #080d14 100%) !important;
    border: 1px solid var(--border-hi);
    border-radius: 20px;
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.4rem;
    display: flex; align-items: center; justify-content: space-between;
    box-shadow: 0 8px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.03);
    position: relative;
    overflow: hidden;
}
.rec-hero::before {
    content: "";
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(94,200,200,0.3) 40%, rgba(94,200,200,0.3) 60%, transparent);
}
.rec-hero * { color: inherit !important; }
.rec-hero-title {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.8rem; font-weight: 800;
    background: linear-gradient(120deg, var(--accent-hi) 0%, var(--accent) 40%, var(--accent-dk) 100%);
    background-size: 200% auto;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0; letter-spacing: -0.03em;
    animation: gradientShift 4s ease infinite;
}
.rec-hero-sub { color: var(--text-dim) !important; font-size: 0.82rem; margin: 5px 0 0; }

/* ── Count badge ──────────────────────────────────────────────────── */
.count-badge {
    background: linear-gradient(135deg, var(--accent), var(--accent-dk)) !important;
    color: #04060c !important; border-radius: 999px;
    padding: 6px 18px; font-size: 0.78rem; font-weight: 800;
    box-shadow: 0 4px 16px rgba(94,200,200,0.35);
    letter-spacing: 0.04em;
}

/* ── Tabs ─────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { background: transparent !important; }
.stTabs [data-baseweb="tab"] { color: var(--text-dim) !important; font-weight: 600 !important; background: transparent !important; }
.stTabs [aria-selected="true"] { color: var(--accent) !important; border-bottom-color: var(--accent) !important; }
.stTabs [data-baseweb="tab-panel"] { background: transparent !important; }

/* ══════════════════════════════════════════════════════════════════
   LANDING PAGE — hero, marquee, sign-in card
   ══════════════════════════════════════════════════════════════════ */

@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes marquee {
    from { transform: translateX(0); }
    to   { transform: translateX(-50%); }
}
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 4px 40px rgba(0,0,0,0.5), 0 0 0 1px var(--border); }
    50%       { box-shadow: 0 4px 40px rgba(0,0,0,0.5), 0 0 40px 2px rgba(94,200,200,0.12), 0 0 0 1px rgba(94,200,200,0.2); }
}
@keyframes dotPulse {
    0%, 100% { opacity: 0.3; }
    50%       { opacity: 0.6; }
}

.fits-landing-hero {
    position: relative;
    overflow: hidden;
    border-radius: 24px;
    padding: 3.8rem 3.2rem 3rem;
    margin-bottom: 0;
    background:
        linear-gradient(145deg, #080c18 0%, #06090f 40%, #08111c 70%, #060a12 100%);
    border: 1px solid var(--border-hi);
    box-shadow: 0 16px 64px rgba(0,0,0,0.7), inset 0 1px 0 rgba(255,255,255,0.04);
}
/* Dot grid overlay */
.fits-landing-hero::after {
    content: "";
    position: absolute;
    inset: 0;
    background-image: radial-gradient(circle, rgba(94,200,200,0.12) 1px, transparent 1px);
    background-size: 28px 28px;
    mask-image: radial-gradient(ellipse 80% 70% at 50% 50%, black 30%, transparent 100%);
    -webkit-mask-image: radial-gradient(ellipse 80% 70% at 50% 50%, black 30%, transparent 100%);
    pointer-events: none;
    animation: dotPulse 6s ease-in-out infinite;
}
/* Radial colour bloom */
.fits-landing-hero::before {
    content: "";
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse 70% 50% at 10% 0%, rgba(94,200,200,0.09) 0%, transparent 60%),
        radial-gradient(ellipse 50% 70% at 90% 100%, rgba(62,168,168,0.07) 0%, transparent 55%),
        radial-gradient(ellipse 30% 30% at 55% 25%, rgba(125,232,232,0.05) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}

.fits-wordmark {
    position: relative; z-index: 1;
    font-size: clamp(4rem, 10vw, 7.5rem);
    font-weight: 900;
    letter-spacing: -0.04em;
    line-height: 0.9;
    margin: 0 0 0.55rem;
    background: linear-gradient(120deg, #ffffff 0%, #c8f0f0 25%, var(--accent) 50%, var(--accent-dk) 75%, #c8f0f0 100%);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradientShift 5s ease infinite, fadeUp 0.8s ease both;
}
.fits-wordmark-sub {
    position: relative; z-index: 1;
    font-size: clamp(0.6rem, 1.4vw, 0.75rem);
    font-weight: 700;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: var(--accent-dk) !important;
    margin-bottom: 1.8rem;
    animation: fadeUp 0.8s cubic-bezier(0.4,0,0.2,1) 0.1s both;
}
.fits-hero-tagline {
    position: relative; z-index: 1;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: clamp(1.25rem, 3vw, 2.1rem);
    font-weight: 700;
    color: var(--text-pri) !important;
    line-height: 1.2;
    max-width: 520px;
    margin-bottom: 0.8rem;
    letter-spacing: -0.02em;
    animation: fadeUp 0.9s cubic-bezier(0.4,0,0.2,1) 0.2s both;
}
.fits-hero-tagline em {
    font-style: normal;
    -webkit-text-fill-color: transparent;
    background: linear-gradient(90deg, var(--accent-hi), var(--accent));
    -webkit-background-clip: text;
    background-clip: text;
}
.fits-hero-sub {
    position: relative; z-index: 1;
    font-size: 0.88rem;
    color: var(--text-dim) !important;
    margin-bottom: 0;
    line-height: 1.6;
    max-width: 480px;
    animation: fadeUp 0.9s cubic-bezier(0.4,0,0.2,1) 0.3s both;
}
.fits-pills-strip {
    position: relative; z-index: 1;
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 1.8rem;
    animation: fadeUp 0.9s cubic-bezier(0.4,0,0.2,1) 0.4s both;
}
.fits-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(94,200,200,0.06);
    border: 1px solid rgba(94,200,200,0.18);
    color: var(--accent) !important;
    border-radius: 999px;
    padding: 5px 14px;
    font-size: 0.69rem;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    transition: background 0.18s ease, border-color 0.18s ease;
}
.fits-pill:hover {
    background: rgba(94,200,200,0.12);
    border-color: rgba(94,200,200,0.3);
}

.fits-marquee-wrap {
    position: relative; z-index: 1;
    overflow: hidden;
    padding: 0.85rem 0;
    border-top: 1px solid rgba(26,34,54,0.8);
    border-bottom: 1px solid rgba(26,34,54,0.8);
    margin: 1.6rem 0 0;
    white-space: nowrap;
    mask-image: linear-gradient(90deg, transparent, black 10%, black 90%, transparent);
    -webkit-mask-image: linear-gradient(90deg, transparent, black 10%, black 90%, transparent);
}
.fits-marquee-inner {
    display: inline-block;
    animation: marquee 24s linear infinite;
}
.fits-marquee-inner span {
    font-size: 0.66rem;
    font-weight: 800;
    letter-spacing: 0.24em;
    text-transform: uppercase;
    color: var(--text-dim) !important;
    padding: 0 1.6rem;
}
.fits-marquee-inner span.hl {
    color: var(--accent) !important;
}

/* ── Sign-in card ─────────────────────────────────────────────────── */
.fits-signin-card {
    background: linear-gradient(160deg, rgba(12,14,22,0.95) 0%, rgba(14,18,32,0.95) 100%);
    border: 1px solid var(--border-hi);
    border-radius: 22px;
    padding: 2.2rem 2.4rem 2.4rem;
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    box-shadow: 0 4px 40px rgba(0,0,0,0.6);
    animation: fadeUp 0.9s cubic-bezier(0.4,0,0.2,1) 0.35s both, pulseGlow 5s ease 3s infinite;
    position: relative;
    overflow: hidden;
}
.fits-signin-fullcard {
    background: linear-gradient(160deg, rgba(12,14,22,0.95) 0%, rgba(14,18,32,0.95) 100%);
    border: 1px solid var(--border-hi);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    box-shadow: 0 4px 40px rgba(0,0,0,0.6);
}
.fits-signin-card::before,
.fits-signin-fullcard::before {
    content: "";
    position: absolute;
    top: 0; left: 10%; right: 10%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(94,200,200,0.5) 50%, transparent);
}
.fits-signin-label {
    font-size: 0.65rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: var(--accent) !important;
    margin-bottom: 1rem;
    opacity: 0.85;
}
.fits-signin-heading {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.4rem;
    font-weight: 800;
    color: var(--text-pri) !important;
    margin-bottom: 0.2rem;
    letter-spacing: -0.03em;
}
.fits-signin-hint {
    font-size: 0.8rem;
    color: var(--text-dim) !important;
    margin-bottom: 1.6rem;
    line-height: 1.5;
}

/* ── Theme toggle ────────────────────────────────────────────────── */
.theme-toggle-wrap {
    display: flex;
    gap: 8px;
    margin-top: 1.2rem;
    justify-content: center;
}
.theme-btn {
    flex: 1;
    padding: 8px 0;
    border-radius: 10px;
    font-size: 0.77rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    cursor: pointer;
    border: 1px solid var(--border-hi);
    background: transparent;
    color: var(--text-dim);
    transition: all 0.18s ease;
}
.theme-btn.active-dark {
    background: rgba(94,200,200,0.06);
    border-color: var(--accent);
    color: var(--accent);
    box-shadow: 0 0 16px rgba(94,200,200,0.1);
}
.theme-btn.active-light {
    background: #f5f0ff;
    border-color: #7c3aed;
    color: #7c3aed;
}
</style>
"""

# ─── Light mode CSS override ──────────────────────────────────────────────────
_FITS_LIGHT_CSS = """
<style>
/*
  LIGHT MODE palette:
    bg-base:    #f8f7ff   (near-white with violet tint)
    bg-card:    #ffffff   (pure white cards)
    bg-glass:   rgba(255,255,255,0.75) (glassmorphism)
    bg-input:   #f0eeff   (soft violet-tinted input)
    border:     #e0d9f7   (soft violet border)
    border-hi:  #c4b5fd   (violet hover border)
    text-pri:   #1a1033   (near-black, deep violet)
    text-sec:   #4b4466   (muted dark violet)
    text-dim:   #7c6fa0   (dim violet)
    accent:     #7c3aed   (vivid violet)
    accent-lt:  #a78bfa   (light violet)
    accent-bg:  #f5f0ff   (tinted violet bg)
*/

html, body, [class*="css"],
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > div,
[data-testid="stAppViewBlockContainer"],
[data-testid="stVerticalBlock"],
[data-testid="stHeader"],
.main, .main > div,
.main .block-container,
.block-container,
[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"],
[data-testid="stToolbar"] {
    color: #1a1033 !important;
    background-color: transparent !important;
}
.stApp, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f8f7ff 0%, #f0eeff 60%, #faf8ff 100%) !important;
}
[data-testid="stAppViewContainer"],
.stApp,
.stApp section[data-testid="stSidebar"] {
    filter: saturate(1.2) !important;
}
[data-testid="stHeader"] {
    background: rgba(248,247,255,0.88) !important;
    backdrop-filter: blur(12px) saturate(1.25) !important;
    border-bottom: 1px solid #e0d9f7 !important;
}
::-webkit-scrollbar-thumb { background: #c4b5fd !important; }
::-webkit-scrollbar-thumb:hover { background: #6b21ff !important; }

/* Sidebar */
.stApp section[data-testid="stSidebar"],
.stApp section[data-testid="stSidebar"] > div,
.stApp section[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, #f5f0ff 0%, #ede9fe 100%) !important;
    border-right: 1px solid #e0d9f7 !important;
}
.stApp section[data-testid="stSidebar"] * { color: #4b4466 !important; background-color: transparent !important; }
.stApp section[data-testid="stSidebar"] h1,
.stApp section[data-testid="stSidebar"] h2,
.stApp section[data-testid="stSidebar"] h3 { color: #1a1033 !important; }
.stApp section[data-testid="stSidebar"] label { color: #7c6fa0 !important; }
.stApp section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"],
.stApp section[data-testid="stSidebar"] .stSelectbox > div > div,
.stApp section[data-testid="stSidebar"] .stSlider > div {
    background: #ede9fe !important;
    border-color: #c4b5fd !important;
}
.stApp section[data-testid="stSidebar"] .stButton > button {
    background: #ffffff !important;
    border: 1px solid #e0d9f7 !important;
    color: #4b4466 !important;
}
.stApp section[data-testid="stSidebar"] .stButton > button:hover {
    border-color: #6b21ff !important;
    color: #6b21ff !important;
    background: #efe5ff !important;
}
section[data-testid="stSidebar"] hr { border-color: #e0d9f7 !important; }
section[data-testid="stSidebar"] .stFormSubmitButton > button {
    background: linear-gradient(135deg, #6b21ff 0%, #5b14eb 100%) !important;
    color: #ffffff !important;
}

/* Main text */
.main h1, .main h2, .main h3,
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3 { color: #1a1033 !important; }
.main p, .main span, .main li, .main label,
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] label { color: #4b4466 !important; }
.main a { color: #6b21ff !important; }

/* Inputs */
.main input, .main textarea,
.main [data-baseweb="select"],
.main [data-baseweb="input"],
.main [data-baseweb="input"] input,
.main [data-baseweb="base-input"],
.main [data-testid="stTextInput"] input,
.main [data-testid="stTextInput"] > div,
.main [data-testid="stTextInput"] > div > div,
.main .stTextInput input, .main .stTextArea textarea,
.main .stNumberInput input,
.main .stSelectbox [data-baseweb="select"],
[data-testid="stAppViewContainer"] input,
[data-testid="stAppViewContainer"] textarea,
[data-testid="stAppViewContainer"] [data-baseweb="select"],
[data-testid="stAppViewContainer"] [data-baseweb="input"],
[data-testid="stAppViewContainer"] [data-baseweb="base-input"],
[data-testid="stAppViewContainer"] [data-testid="stTextInput"] > div,
[data-testid="stAppViewContainer"] [data-testid="stTextInput"] > div > div {
    background-color: #f0eeff !important;
    color: #1a1033 !important;
    border: 1.5px solid #c4b5fd !important;
    caret-color: #7c3aed !important;
}
.main input:focus, .main textarea:focus,
.main [data-baseweb="input"]:focus-within,
[data-testid="stAppViewContainer"] input:focus,
[data-testid="stAppViewContainer"] textarea:focus,
[data-testid="stAppViewContainer"] [data-baseweb="input"]:focus-within {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.15) !important;
}
.main input::placeholder, .main textarea::placeholder { color: #c4b5fd !important; }
.main [data-testid="stTextInput"] label,
.main .stTextInput label, .main .stTextArea label,
.main .stNumberInput label, .main .stSelectbox label,
[data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] { color: #7c6fa0 !important; }

/* Dropdowns */
[data-baseweb="popover"], [data-baseweb="menu"],
[data-baseweb="popover"] ul, [data-baseweb="menu"] ul {
    background-color: #ffffff !important;
    border: 1px solid #e0d9f7 !important;
}
[data-baseweb="popover"] li, [data-baseweb="menu"] li {
    background-color: #ffffff !important;
    color: #1a1033 !important;
}
[data-baseweb="menu"] li:hover, [data-baseweb="popover"] li:hover { background-color: #f5f0ff !important; }

/* Form wrapper */
.main [data-testid="stForm"],
[data-testid="stAppViewContainer"] [data-testid="stForm"] {
    background: rgba(255,255,255,0.8) !important;
    border: 1px solid #e0d9f7 !important;
    box-shadow: 0 4px 24px rgba(124,58,237,0.08) !important;
    backdrop-filter: blur(8px) !important;
}

/* Radio / checkbox */
.main [data-testid="stRadio"] label,
.main [data-testid="stCheckbox"] label { color: #4b4466 !important; }

/* Radio indicator — unselected: hollow violet border */
.stApp [role="radio"] > div:first-child {
    background-color: transparent !important;
    border-color: #c4b5fd !important;
    box-shadow: none !important;
}
/* Radio indicator — selected: solid violet fill */
.stApp [role="radio"][aria-checked="true"] > div:first-child {
    background-color: #7c3aed !important;
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.2) !important;
}
/* Inner dot on selected (base-web places a white dot inside) */
.stApp [role="radio"][aria-checked="true"] > div:first-child > div {
    background-color: #ffffff !important;
}

/* Checkbox checked */
.stApp [data-baseweb="checkbox"] [data-checked="true"] > div,
.stApp [data-baseweb="checkbox"] [aria-checked="true"] > div { background-color: #7c3aed !important; border-color: #7c3aed !important; }

/* Slider thumb */
.stApp [data-testid="stSlider"] [role="slider"] {
    background-color: #7c3aed !important;
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 4px rgba(124,58,237,0.18) !important;
}
/* Slider: empty track (full width bg) */
.stApp [data-testid="stSlider"] [data-baseweb="slider"] > div:first-child {
    background-color: #e0d9f7 !important;
}
/* Slider: filled track (positioned div inside track) */
.stApp [data-testid="stSlider"] [data-baseweb="slider"] > div:first-child > div {
    background-color: #7c3aed !important;
}
/* Fallback class-name based selectors */
.stApp [data-testid="stSlider"] div[class*="Track"] { background-color: #e0d9f7 !important; }
.stApp [data-testid="stSlider"] div[class*="Fill"],
.stApp [data-testid="stSlider"] div[class*="fill"] { background-color: #7c3aed !important; }

/* Multiselect tags */
.main [data-baseweb="tag"] {
    background-color: rgba(124,58,237,0.1) !important;
    border-color: #c4b5fd !important;
    color: #7c3aed !important;
}
.main [data-baseweb="tag"] span { color: #7c3aed !important; }
.main [data-baseweb="tag"] [data-testid="stMultiSelectDeleteButton"] { color: #7c3aed !important; }

/* Buttons */
.stApp .main .stButton > button,
.stApp [data-testid="stAppViewContainer"] .stButton > button {
    border: 1.5px solid #e0d9f7 !important;
    background: #ffffff !important;
    color: #4b4466 !important;
    box-shadow: 0 1px 3px rgba(124,58,237,0.08) !important;
}
.stApp .main .stButton > button:hover,
.stApp [data-testid="stAppViewContainer"] .stButton > button:hover {
    border-color: #7c3aed !important;
    color: #7c3aed !important;
    background: #f5f0ff !important;
    box-shadow: 0 2px 8px rgba(124,58,237,0.15) !important;
}
.stApp .main div[data-testid="stButton"] button[kind="primary"],
.stApp .main .stFormSubmitButton > button,
.stApp [data-testid="stAppViewContainer"] .stFormSubmitButton > button {
    background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 2px 8px rgba(124,58,237,0.25) !important;
}
.stApp .main div[data-testid="stButton"] button[kind="primary"]:hover,
.stApp .main .stFormSubmitButton > button:hover,
.stApp [data-testid="stAppViewContainer"] .stFormSubmitButton > button:hover {
    background: linear-gradient(135deg, #6d28d9 0%, #5b21b6 100%) !important;
    box-shadow: 0 4px 16px rgba(124,58,237,0.3) !important;
}
.stApp .main .stDownloadButton > button {
    background: #ffffff !important;
    border: 1.5px solid #e0d9f7 !important;
    color: #4b4466 !important;
}

/* Sidebar buttons — light */
.stApp section[data-testid="stSidebar"] .stButton > button {
    background: #ffffff !important;
    border: 1px solid #e0d9f7 !important;
    color: #4b4466 !important;
}
.stApp section[data-testid="stSidebar"] .stButton > button:hover {
    border-color: #7c3aed !important;
    color: #7c3aed !important;
    background: #f5f0ff !important;
}

/* Alerts */
.main [data-testid="stAlert"],
.main div[data-baseweb="notification"] {
    background: rgba(255,255,255,0.9) !important;
    border: 1px solid #e0d9f7 !important;
}

/* Progress bar */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #7c3aed, #a78bfa) !important;
}
.stProgress > div > div { background: #ede9fe !important; }

/* Expanders */
.main details, [data-testid="stExpander"] {
    background: rgba(255,255,255,0.8) !important;
    border: 1px solid #e0d9f7 !important;
    backdrop-filter: blur(6px) !important;
}
.main details summary span { color: #4b4466 !important; }

/* Captions / dividers */
.main [data-testid="stCaptionContainer"] span { color: #7c6fa0 !important; }
.main hr { border-color: #e0d9f7 !important; }

/* Metrics */
[data-testid="stMetricValue"] { color: #1a1033 !important; }
[data-testid="stMetricLabel"] { color: #7c6fa0 !important; }

/* Tabs */
.stTabs [data-baseweb="tab"] { color: #7c6fa0 !important; }
.stTabs [aria-selected="true"] { color: #7c3aed !important; border-bottom-color: #7c3aed !important; }

/* Spinner */
.stSpinner > div { border-top-color: #7c3aed !important; }

/* Toast */
[data-testid="stToast"] {
    background: #ffffff !important;
    color: #1a1033 !important;
    border: 1px solid #e0d9f7 !important;
}

/* Product cards */
.product-card {
    background: rgba(255,255,255,0.85) !important;
    border: 1px solid #e0d9f7 !important;
    box-shadow: 0 2px 12px rgba(124,58,237,0.07) !important;
    backdrop-filter: blur(6px) !important;
}
.product-card:hover { border-color: #c4b5fd !important; box-shadow: 0 8px 32px rgba(124,58,237,0.13) !important; }
.product-brand { color: #7c3aed !important; }
.product-title { color: #1a1033 !important; }
.score-pill {
    background: rgba(124,58,237,0.08) !important;
    color: #7c3aed !important;
    border: 1px solid rgba(124,58,237,0.2) !important;
}
.filter-pill {
    background: rgba(124,58,237,0.07) !important;
    color: #7c3aed !important;
    border: 1px solid rgba(124,58,237,0.2) !important;
}
.rec-hero {
    background: linear-gradient(135deg, #f5f0ff 0%, #ede9fe 60%, #f8f7ff 100%) !important;
    border: 1px solid #e0d9f7 !important;
    box-shadow: 0 4px 24px rgba(124,58,237,0.08) !important;
}
.rec-hero-title {
    background: linear-gradient(90deg, #7c3aed, #a78bfa) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
}
.rec-hero-sub { color: #7c6fa0 !important; }
.count-badge {
    background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
    color: #ffffff !important;
}

/* Landing hero in light mode */
.fits-landing-hero {
    background: linear-gradient(135deg, #f5f0ff 0%, #ede9fe 40%, #e9e3ff 100%) !important;
    border: 1px solid #e0d9f7 !important;
    box-shadow: 0 8px 48px rgba(124,58,237,0.12) !important;
}
.fits-landing-hero::before {
    background:
        radial-gradient(ellipse 80% 60% at 20% 10%, rgba(124,58,237,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 80% at 85% 90%, rgba(109,40,217,0.06) 0%, transparent 55%) !important;
}
.fits-wordmark {
    background: linear-gradient(120deg, #1a1033 0%, #7c3aed 40%, #a78bfa 70%, #6d28d9 100%) !important;
    background-size: 300% 300% !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
}
.fits-wordmark-sub { color: #7c3aed !important; }
.fits-hero-tagline { color: #1a1033 !important; }
.fits-hero-tagline em {
    background: linear-gradient(90deg, #7c3aed, #a78bfa) !important;
    -webkit-background-clip: text !important;
    background-clip: text !important;
}
.fits-hero-sub { color: #7c6fa0 !important; }
.fits-pill {
    background: rgba(124,58,237,0.08) !important;
    border: 1px solid rgba(124,58,237,0.22) !important;
    color: #7c3aed !important;
}
.fits-marquee-wrap { border-color: #e0d9f7 !important; }
.fits-marquee-inner span { color: #c4b5fd !important; }
.fits-marquee-inner span.hl { color: #7c3aed !important; }

/* Sign-in card in light mode */
.fits-signin-fullcard, .fits-signin-card {
    background: rgba(255,255,255,0.85) !important;
    border: 1px solid #e0d9f7 !important;
    box-shadow: 0 4px 32px rgba(124,58,237,0.1) !important;
    backdrop-filter: blur(10px) !important;
}
.fits-signin-fullcard::before, .fits-signin-card::before {
    background: linear-gradient(90deg, transparent, #7c3aed 50%, transparent) !important;
}
.fits-signin-label { color: #7c3aed !important; }
.fits-signin-heading { color: #1a1033 !important; }
.fits-signin-hint { color: #7c6fa0 !important; }
</style>
"""

# ─── Comfort mode CSS override ───────────────────────────────────────────────
_FITS_COMFORT_CSS = """
<style>
/*
  COMFORT MODE palette (older/low-quality displays):
    bg-base:    #141517
    bg-card:    #1b1d21
    bg-input:   #101214
    border:     #32353b
    text-pri:   #ece8dd
    text-sec:   #d3cec2
    text-dim:   #aaa797
    accent:     #9fae9f
    accent-hi:  #bcc8bc
*/

.stApp {
    --bg: #141517 !important;
    --bg-raised: #1a1b1f !important;
    --bg-card: #1b1d21 !important;
    --bg-input: #101214 !important;
    --border: #32353b !important;
    --border-hi: #4a4f58 !important;
    --text-pri: #ece8dd !important;
    --text-sec: #d3cec2 !important;
    --text-dim: #aaa797 !important;
    --accent: #b2d878 !important;
    --accent-hi: #d6f39f !important;
    --accent-dk: #95c04d !important;
}

.stApp, [data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #141517 0%, #17191c 100%) !important;
}

[data-testid="stAppViewContainer"],
.stApp,
.stApp section[data-testid="stSidebar"] {
    filter: saturate(1.2) !important;
}

[data-testid="stHeader"] {
    background: rgba(20,21,23,0.92) !important;
    border-bottom: 1px solid #30333a !important;
    backdrop-filter: blur(10px) saturate(1.18) !important;
}

.stApp section[data-testid="stSidebar"],
.stApp section[data-testid="stSidebar"] > div,
.stApp section[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, #17191c 0%, #15171a 100%) !important;
}

/* Lower visual noise */
.fits-landing-hero::after,
.fits-landing-hero::before,
.rec-hero::before,
.product-card::before {
    opacity: 0.22 !important;
}

/* Disable flashy animation in comfort mode */
.fits-wordmark,
.fits-wordmark-sub,
.fits-hero-tagline,
.fits-hero-sub,
.fits-pills-strip,
.fits-marquee-inner,
.fits-signin-card,
.fits-signin-fullcard,
.rec-hero-title,
.stProgress > div > div > div {
    animation: none !important;
}

.product-card:hover,
.main .stButton > button:hover,
[data-testid="stAppViewContainer"] .stButton > button:hover,
.main div[data-testid="stButton"] button[kind="primary"]:hover,
.main .stFormSubmitButton > button:hover {
    transform: none !important;
}

/* Slightly larger text for weak panels */
.main p,
.main span,
.main li,
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] li {
    font-size: 0.97rem !important;
    line-height: 1.55 !important;
}

.main [data-testid="stWidgetLabel"],
.main [data-testid="stWidgetLabel"] span,
[data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] span {
    font-size: 0.8rem !important;
    color: #c9c4b7 !important;
}

.main [data-testid="stForm"],
[data-testid="stAppViewContainer"] [data-testid="stForm"] {
    backdrop-filter: none !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.28) !important;
}

.fits-marquee-wrap { display: none !important; }

.main .stButton > button,
[data-testid="stAppViewContainer"] .stButton > button {
    border-width: 1.7px !important;
}
</style>
"""

# ─── Fullscreen CSS ───────────────────────────────────────────────────────────
_FITS_FULLSCREEN_CSS = """
<style>
[data-testid="stHeader"]         { display: none !important; }
[data-testid="stToolbar"]        { display: none !important; }
[data-testid="stDecoration"]     { display: none !important; }
.main .block-container {
    padding-top: 1rem !important;
    max-width: 100% !important;
}
section[data-testid="stSidebar"] { top: 0 !important; }
</style>
"""


def main():
    st.set_page_config(
        page_title="FITS — Fashion Intelligence",
        page_icon="✦",
        layout="wide",
    )
    st.markdown(_FITS_GLOBAL_CSS, unsafe_allow_html=True)
    _init_state()
    _theme = st.session_state.get("theme", "dark")
    if _theme == "light":
        st.markdown(_FITS_LIGHT_CSS, unsafe_allow_html=True)
    elif _theme == "comfort":
        st.markdown(_FITS_COMFORT_CSS, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        _is_light = _theme == "light"
        _is_comfort = _theme == "comfort"
        _accent   = "#7c3aed" if _is_light else ("#9fae9f" if _is_comfort else "#6cc8c8")
        _sub_col  = "#7c6fa0" if _is_light else ("#b3ad9f" if _is_comfort else "#8b949e")
        st.markdown(
            f"<h2 style='color:{_accent};font-weight:800;margin-bottom:0'>✦ FITS</h2>"
            f"<p style='color:{_sub_col};font-size:0.78rem;margin-top:2px'>Fashion Intelligence & Recommendation</p>",
            unsafe_allow_html=True,
        )
        st.divider()

        st.divider()

        def _reset_app_state():
            for k in list(st.session_state.keys()):
                del st.session_state[k]

        st.button("🔄 Start Over", use_container_width=True, on_click=_reset_app_state)

        # ── Dev tools (collapse by default) ──────────────────────────────────
        with st.expander("🛠 Dev", expanded=False):
            st.caption("Skip login & jump to any step")

            def _dev_login():
                st.session_state.user_id   = "dev-user-0000"
                st.session_state.user_name = "Dev User"

            def _dev_jump(_si):
                if not st.session_state.get("user_id"):
                    st.session_state.user_id   = "dev-user-0000"
                    st.session_state.user_name = "Dev User"
                # Seed sensible defaults so recommendation/results pages don't crash
                if _si >= 4 and not st.session_state.get("final_attrs"):
                    st.session_state.final_attrs = {
                        "face_shape":   "oval",
                        "body_type":    "rectangle",
                        "skin_color":   "#c68642",
                        "eyes_color":   "#634e34",
                        "lips_color":   "#c0706a",
                        "hair_color":   "#3b2314",
                        "hair_length":  "medium",
                        "undertone":    "warm",
                        "neck_length":  "average",
                        "season":       "autumn",
                    }
                if _si >= 2 and not st.session_state.get("body_type"):
                    st.session_state.body_type = "rectangle"
                st.session_state.step = _si

            if not st.session_state.get("user_id"):
                st.button("⚡ Dev login", use_container_width=True, key="dev_login", on_click=_dev_login)
            else:
                st.caption(f"Logged in as **{st.session_state.get('user_name','?')}**")
            st.divider()
            _step_labels = [f"{i}: {s}" for i, s in enumerate(STEPS)]
            for _si, _sl in enumerate(_step_labels):
                st.button(_sl, use_container_width=True, key=f"dev_step_{_si}", on_click=_dev_jump, args=(_si,))

    # ── Top step progress bar ─────────────────────────────────────────────────
    _cur = st.session_state.step
    _tot = len(STEPS)
    _bar_accent  = _theme_color("#7c3aed", "#6cc8c8", "#9fae9f")
    _bar_track   = _theme_color("#e0d9f7", "#21262d", "#2b2e34")
    _bar_txt     = _theme_color("#7c3aed", "#6cc8c8", "#b8b09b")
    _bar_sub     = _theme_color("#7c6fa0", "#8b949e", "#aaa797")
    _pct         = round(_cur / (_tot - 1) * 100, 1) if _tot > 1 else 0
    _step_name   = STEPS[_cur] if _cur < _tot else STEPS[-1]
    st.markdown(
        f"<div style='margin-bottom:1rem'>"
        f"<div style='display:flex;justify-content:space-between;align-items:baseline;margin-bottom:5px'>"
        f"<span style='font-size:0.72rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:{_bar_txt}'>{_step_name}</span>"
        f"<span style='font-size:0.68rem;color:{_bar_sub}'>Step {_cur + 1} of {_tot}</span>"
        f"</div>"
        f"<div style='width:100%;height:5px;border-radius:99px;background:{_bar_track};overflow:hidden'>"
        f"<div style='height:100%;width:{_pct}%;border-radius:99px;background:{_bar_accent};"
        f"transition:width 0.4s ease'></div>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Step router ───────────────────────────────────────────────────────────
    step = st.session_state.step
    if step == 0:
        _step_identity()
    elif step == 1:
        _step_capture()
    elif step == 2:
        _step_body_type()
    elif step == 3:
        _step_analysis()
    elif step == 4:
        _step_results()
    elif step == 5:
        _step_recommendations()
    elif step == 6:
        _step_tryon()


# ═══════════════════════════════════════════════════════════════════════════════
# DB AUTO-SAVE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _build_product_feedback(results: list) -> list:
    """Build per-product feedback list: each product gets its score + like/dislike/null."""
    feedback_map = st.session_state.feedback  # {product_id: "like"|"dislike"}
    return [
        {
            "id":       r.get("id", ""),
            "name":     r["name"],
            "brand":    r.get("brand", ""),
            "category": r.get("category", ""),
            "score":    round(r.get("adjusted_score", r.get("score", 0)), 3),
            "feedback": feedback_map.get(r.get("id")),  # "like", "dislike", or None
        }
        for r in results
    ]


def _refresh_products_global():
    """Rebuild products_global.json + profiles_new_quiz.json from current data.
    Runs in a daemon background thread so it never blocks the UI."""
    try:
        import csv as _csv
        import json as _json
        import threading as _threading
        from collections import defaultdict as _defaultdict
        from pathlib import Path as _P

        def _run():
            _db        = _P(__file__).resolve().parent / "TASK_1" / "db"
            _csv_path  = _P(__file__).resolve().parent / "TASK_2" / "FITS_Recommender" / "FITS_Recommender" / "products.csv"
            _exp_path  = _db / "experiments.json"
            _out_path  = _db / "products_global.json"
            _new_path  = _db / "profiles_new_quiz.json"
            _prof_path = _db / "profiles.json"

            NEW_QUIZ_KEYS = {"vibe", "shopping_mode", "adventurousness", "priority", "buy_frequency"}

            with open(_csv_path, encoding="utf-8") as f:
                catalog = {row["product_id"]: row for row in _csv.DictReader(f)}

            with open(_exp_path, encoding="utf-8") as f:
                exp_data = _json.load(f)
            experiments = exp_data["experiments"]

            new_quiz_exps = [
                e for e in experiments
                if set(e.get("quiz_answers", {}).keys()) & NEW_QUIZ_KEYS
            ]

            likes    = _defaultdict(int)
            dislikes = _defaultdict(int)
            shown    = _defaultdict(int)
            for exp in new_quiz_exps:
                for item in exp.get("ranked_items", []) + exp.get("discovery_items", []):
                    pid = str(item["id"]).zfill(6)
                    shown[pid] += 1
                    fb = item.get("feedback")
                    if fb == "like":       likes[pid]    += 1
                    elif fb == "dislike":  dislikes[pid] += 1

            products = []
            for pid, row in catalog.items():
                products.append({
                    "product_id":    pid,
                    "name":          row["clean_name"],
                    "brand":         row["brand"],
                    "gender":        row["gender"],
                    "category":      row["category_name"],
                    "sleeve_type":   row["sleeve_type"],
                    "neckline_type": row["neckline_type"],
                    "color":         row["color"],
                    "price":         row["price"],
                    "currency":      row["currency"],
                    "description":   row["description"],
                    "image_path":    row["image_path"],
                    "url":           row["url"],
                    "dress_length":  row["dress_length"],
                    "jeans_type":    row["jeans_type"],
                    "pants_type":    row["pants_type"],
                    "feedback": {
                        "likes":    likes.get(pid, 0),
                        "dislikes": dislikes.get(pid, 0),
                        "shown":    shown.get(pid, 0),
                    },
                })
            products.sort(key=lambda x: (-x["feedback"]["likes"], x["product_id"]))

            out = {
                "_meta": {
                    "description":            "Global product catalog — feedback from new-quiz experiments only",
                    "total_products":         len(products),
                    "total_experiments":      len(experiments),
                    "new_quiz_experiments":   len(new_quiz_exps),
                    "products_with_likes":    sum(1 for p in products if p["feedback"]["likes"] > 0),
                    "products_with_dislikes": sum(1 for p in products if p["feedback"]["dislikes"] > 0),
                },
                "products": products,
            }
            with open(_out_path, "w", encoding="utf-8") as f:
                _json.dump(out, f, indent=2, ensure_ascii=False)

            # Also rebuild profiles_new_quiz.json
            with open(_prof_path, encoding="utf-8") as f:
                prof_data = _json.load(f)
            new_uids = {e["user_id"] for e in new_quiz_exps}
            new_prof = {uid: prof_data["profiles"][uid] for uid in new_uids if uid in prof_data["profiles"]}
            new_quiz_out = {
                "_meta": {
                    "description":       "Profiles + experiments using the new 5-question quiz schema",
                    "quiz_schema":       "new",
                    "quiz_keys":         sorted(NEW_QUIZ_KEYS),
                    "total_profiles":    len(new_prof),
                    "total_experiments": len(new_quiz_exps),
                },
                "profiles":    new_prof,
                "experiments": new_quiz_exps,
            }
            with open(_new_path, "w", encoding="utf-8") as f:
                _json.dump(new_quiz_out, f, indent=2, ensure_ascii=False)

        _threading.Thread(target=_run, daemon=True).start()
    except Exception:
        pass  # never crash the app over a background rebuild


def _autosave_rec_session(uid: str, results: list):
    """Create or replace the current rec_session under the user's analysis session."""
    if not results:
        return
    
    # Start new experiment if not already tracking one
    if not st.session_state.get("current_experiment_id"):
        exp_repo = get_experiment_repository()
        exp_id, rec_sid = exp_repo.start_experiment(
            uid,
            filters={k: v for k, v in {
                "occasion": st.session_state.get("_last_occasion", ""),
                "clothing_type": st.session_state.get("_last_clothing", ""),
                "gender": st.session_state.get("_last_gender", ""),
                "skin_tone": st.session_state.get("_last_skin_tone", ""),
                "body_type": st.session_state.get("_last_body_type", ""),
                "face_shape": st.session_state.get("_last_face_shape", ""),
            }.items() if v},
            quiz_answers=st.session_state.get("style_quiz", {}),
        )
        st.session_state.current_experiment_id = exp_id
    
    # Separate ranked and discovery items
    ranked = [r for r in results if r.get("source") == "ranked"]
    discovery = [r for r in results if r.get("source") == "random"]

    # Save all recommendations to experiment (source of truth for item data + feedback)
    exp_repo = get_experiment_repository()
    exp_repo.save_recommendations(
        st.session_state.current_experiment_id,
        ranked,
        discovery,
    )
    # Rebuild products_global.json once per session (not on every rec call)
    if not st.session_state.get('_products_global_refreshed'):
        _refresh_products_global()
        st.session_state['_products_global_refreshed'] = True

    active_filters = {k: v for k, v in {
        "occasion":      st.session_state.get("_last_occasion", ""),
        "clothing_type": st.session_state.get("_last_clothing", ""),
        "gender":        st.session_state.get("_last_gender", ""),
        "skin_tone":     st.session_state.get("_last_skin_tone", ""),
        "body_type":     st.session_state.get("_last_body_type", ""),
        "face_shape":    st.session_state.get("_last_face_shape", ""),
    }.items() if v}
    # profiles.json stores a lightweight reference only — experiments.json is the full source of truth
    ranked_ref = [
        {"id": r.get("id",""), "name": r["name"], "brand": r.get("brand",""),
         "category": r.get("category",""),
         "score": round(r.get("adjusted_score", r.get("score", 0)), 3),
         "source": r.get("source", "ranked")}
        for r in results
    ]
    analysis_sid = st.session_state.get("current_session_id")
    rec_sid = get_repository().save_rec_session(
        uid, active_filters, ranked_ref, [], analysis_sid  # product_feedback tracked in experiments.json
    )
    st.session_state.current_rec_session_id = rec_sid


def _autoupdate_feedback(uid: str):
    """Patch product_feedback + updated_at on the current rec_session in the DB."""
    rec_sid = st.session_state.get("current_rec_session_id")
    if not rec_sid:
        return
    results = st.session_state.get("rec_results") or []
    product_feedback = _build_product_feedback(results)
    get_repository().update_product_feedback(uid, rec_sid, product_feedback)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 0 — Identity
# ═══════════════════════════════════════════════════════════════════════════════

def _step_identity():
    marquee_items = (
        '<span>Style Intelligence</span><span class="hl">✦</span>'
        '<span>AI Analysis</span><span class="hl">✦</span>'
        '<span>Personalized Picks</span><span class="hl">✦</span>'
        '<span>Virtual Try-On</span><span class="hl">✦</span>'
        '<span>Colour Matching</span><span class="hl">✦</span>'
        '<span>Body-Type Fit</span><span class="hl">✦</span>'
        '<span>Brand Discovery</span><span class="hl">✦</span>'
        '<span>Curated For You</span><span class="hl">✦</span>'
    )

    def _render_post_continue_options():
        def _go_use_last():
            st.session_state.final_attrs = st.session_state.returning_attrs
            st.session_state.step = 5

        def _go_new_analysis():
            st.session_state.returning_attrs = None
            st.session_state.step = 1

        def _go_know_attrs():
            st.session_state.step = 5

        def _go_start_analysis_new_profile():
            st.session_state.new_profile = False
            st.session_state.step = 1

        def _go_know_attrs_new_profile():
            st.session_state.new_profile = False
            st.session_state.step = 5

        if st.session_state.get("user_id") and st.session_state.get("returning_attrs"):
            name = st.session_state.user_name
            last_dt = st.session_state.last_session_at or "unknown date"
            now = datetime.now().strftime("%B %d, %Y  %I:%M %p")
            st.success(f"Welcome back, **{name}**! 🕐 {now}  |  Last session: {last_dt}")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.button("📋 Use Last Results + Recommendations", use_container_width=True, key="ret_use_last", on_click=_go_use_last)
            with c2:
                st.button("📷 New Analysis", use_container_width=True, key="ret_new_analysis", on_click=_go_new_analysis)
            with c3:
                st.button("🎯 I Know My Attributes", use_container_width=True, key="ret_know_attrs", on_click=_go_know_attrs)
        elif st.session_state.get("user_id"):
            if st.session_state.get("new_profile"):
                st.success(f"Profile created! Welcome, **{st.session_state.user_name}** 👋")
            else:
                st.info(f"Welcome, **{st.session_state.user_name}**! No previous analysis found.")
            nc1, nc2 = st.columns(2)
            with nc1:
                st.button("📷 Start Analysis", use_container_width=True, type="primary", key="new_start_analysis", on_click=_go_start_analysis_new_profile)
            with nc2:
                st.button("🎯 I Know My Attributes", use_container_width=True, key="new_know_attrs", on_click=_go_know_attrs_new_profile)

    # ── Equal-height columns via CSS ──────────────────────────────────────────
    st.markdown("""
        <style>
        div[data-testid="stHorizontalBlock"]:has(.fits-landing-hero) {
            align-items: stretch !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.fits-landing-hero)
            > div[data-testid="stColumn"] {
            display: flex !important;
            flex-direction: column !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.fits-landing-hero)
            > div[data-testid="stColumn"]
            > div[data-testid="stVerticalBlock"] {
            flex: 1 !important;
            display: flex !important;
            flex-direction: column !important;
        }
        .fits-landing-hero { height: 100% !important; box-sizing: border-box; }
        .fits-signin-fullcard {
            height: 100%;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            justify-content: center;
            border-radius: 18px;
            padding: 1.35rem 1.65rem 1.45rem;
            position: relative;
            overflow: hidden;
            animation: fadeUp 1s ease 0.35s both, pulseGlow 4s ease 2s infinite;
        }
        .fits-signin-heading { margin-bottom: 0.12rem !important; line-height: 1.18 !important; }
        .fits-signin-hint { margin-bottom: 0 !important; line-height: 1.45 !important; }

        /* Sign-in column rhythm */
        div[data-testid="stColumn"]:has(.fits-signin-fullcard) [data-testid="stForm"] {
            margin-top: 0.55rem !important;
            padding: 1.25rem 1.5rem 1.35rem !important;
            border-radius: 18px !important;
        }
        div[data-testid="stColumn"]:has(.fits-signin-fullcard)
            [data-testid="stForm"] [data-testid="stFormSubmitButton"] {
            margin-top: 0.5rem !important;
        }
        div[data-testid="stColumn"]:has(.fits-signin-fullcard)
            > div[data-testid="stVerticalBlock"]
            > div[data-testid="stElementContainer"] {
            margin-bottom: 0.45rem !important;
        }
        div[data-testid="stColumn"]:has(.fits-signin-fullcard) div[data-testid="stHorizontalBlock"] {
            margin-top: 0.15rem !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # ── Hero + Sign-in side by side ───────────────────────────────────────────
    hero_col, signin_col = st.columns([3, 2], gap="large", vertical_alignment="top")

    with hero_col:
        st.markdown(
            f"""
            <div class="fits-landing-hero" style="height:100%">
              <div class="fits-wordmark">FITS</div>
              <div class="fits-wordmark-sub">Fashion Intelligence &amp; Recommendation System</div>
              <div class="fits-hero-tagline">Dress like you <em>mean it.</em><br>Recommended by&nbsp;us, worn&nbsp;by&nbsp;you.</div>
              <div class="fits-hero-sub">Upload a photo &mdash; we analyse your features, match your vibe, and surface the exact pieces that work for your body, skin tone, and style.</div>
              <div class="fits-pills-strip">
                <span class="fits-pill">⚡ AI body &amp; face analysis</span>
                <span class="fits-pill">🎨 Colour-matched recommendations</span>
                <span class="fits-pill">👗 Virtual try-on</span>
                <span class="fits-pill">✦ 3 000+ curated products</span>
              </div>
              <div class="fits-marquee-wrap">
                <div class="fits-marquee-inner">
                  {marquee_items * 2}
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with signin_col:
        st.markdown('<div class="fits-signin-fullcard"><div class="fits-signin-label">✦ Get Started</div><div class="fits-signin-heading">Your wardrobe, upgraded.</div><div class="fits-signin-hint">Sign in or create a profile — takes 10 seconds.</div></div>', unsafe_allow_html=True)
        with st.form("identity_form"):
            name  = st.text_input("Full name", placeholder="e.g. Alex Johnson")
            email = st.text_input("E-mail", placeholder="you@example.com")
            sub   = st.form_submit_button("Continue →", use_container_width=True)
        # Theme toggle directly below the form
        _cur_theme = st.session_state.get("theme", "dark")
        lp_t1, lp_t2, lp_t3 = st.columns(3)

        def _set_theme_dark():
            st.session_state.theme = "dark"

        def _set_theme_light():
            st.session_state.theme = "light"

        def _set_theme_comfort():
            st.session_state.theme = "comfort"

        with lp_t1:
            st.button("🌙 Dark", use_container_width=True, key="lp_theme_dark", on_click=_set_theme_dark)
        with lp_t2:
            st.button("☀️ Light", use_container_width=True, key="lp_theme_light", on_click=_set_theme_light)
        with lp_t3:
            st.button("🟫 Comfort", use_container_width=True, key="lp_theme_comfort", on_click=_set_theme_comfort)

        _theme_hint = {
            "dark": "Theme: Dark",
            "light": "Theme: Light",
            "comfort": "Theme: Comfort (low glare)",
        }
        st.caption(_theme_hint.get(_cur_theme, "Theme: Dark"))

        with st.container():
            _fs_theme = st.session_state.get("theme", "dark")
            if _fs_theme == "light":
                _fs_bg = "#7c3aed"
                _fs_fg = "#ffffff"
            elif _fs_theme == "comfort":
                _fs_bg = "#9fae9f"
                _fs_fg = "#111315"
            else:
                _fs_bg = "#6cc8c8"
                _fs_fg = "#0d1117"
            _st_components.html(f"""
<style>
  body{{margin:0;padding:0;background:transparent}}
  #fsBtn{{
    width:100%;height:38px;border:none;border-radius:6px;
    background:{_fs_bg};color:{_fs_fg};
    font-size:0.85rem;font-weight:600;cursor:pointer;
    font-family:sans-serif;
  }}
  #fsBtn:hover{{opacity:0.88}}
</style>
<button id="fsBtn">⛶ Fullscreen</button>
<script>
(function(){{
  var btn = document.getElementById('fsBtn');
  var d   = window.parent.document;

  function updateLabel(){{
    btn.textContent = d.fullscreenElement ? '⛶ Exit full' : '⛶ Fullscreen';
  }}
  d.addEventListener('fullscreenchange',       updateLabel);
  d.addEventListener('webkitfullscreenchange', updateLabel);

  function injectCSS(){{
    if (d.getElementById('fits-fs-style')) return;
    var s = d.createElement('style');
    s.id = 'fits-fs-style';
    s.textContent = [
      ':fullscreen header[data-testid="stHeader"]{{display:none!important}}',
      ':fullscreen .stToolbar{{display:none!important}}',
      ':-webkit-full-screen header[data-testid="stHeader"]{{display:none!important}}',
      ':-webkit-full-screen .stToolbar{{display:none!important}}'
    ].join('');
    d.head.appendChild(s);
  }}

  btn.addEventListener('click', function(){{
    injectCSS();
    if (!d.fullscreenElement){{
      (d.documentElement.requestFullscreen || d.documentElement.webkitRequestFullscreen)
        .call(d.documentElement);
    }} else {{
      (d.exitFullscreen || d.webkitExitFullscreen).call(d);
    }}
  }});
}})();
</script>
""", height=46)

    _render_post_continue_options()

    if sub:
        name  = name.strip()
        email = email.strip().lower()
        if not name or not email:
            st.error("Both fields are required.")
            return
        repo    = get_repository()
        profile = repo.find_by_email(email)
        if profile is None:
            profile = repo.create_profile(name, email)
            st.session_state.user_id    = profile.user_id
            st.session_state.user_name  = profile.name
            st.session_state.new_profile = True
            st.rerun()
        else:
            last_attrs = repo.get_latest_attributes(profile.user_id)
            raw_dt     = repo.get_last_run_at(profile.user_id)
            last_dt    = None
            if raw_dt:
                try:
                    parsed  = datetime.fromisoformat(raw_dt.replace("Z", "+00:00"))
                    last_dt = parsed.strftime("%B %d, %Y  %I:%M %p")
                except Exception:
                    last_dt = raw_dt[:10] if raw_dt else None
            st.session_state.user_id        = profile.user_id
            st.session_state.user_name      = profile.name
            st.session_state.returning_attrs = last_attrs
            st.session_state.last_session_at = last_dt
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Body Type
# ═══════════════════════════════════════════════════════════════════════════════

def _step_body_type():
    _sub_col  = _theme_color("#7c6fa0", "#8b949e", "#b3ad9f")
    _name_col = _theme_color("#1a1033", "#e6edf3", "#ece8dd")
    st.markdown(
        "<h1 style='font-size:2rem;font-weight:800;letter-spacing:-0.02em;margin-bottom:0'>"
        "Step 2 — Body Type</h1>"
        f"<p style='color:{_sub_col};font-size:0.95rem;margin-bottom:1rem'>"
        f"Hi <b style='color:{_name_col}'>{st.session_state.user_name}</b>, which body shape fits you best?</p>",
        unsafe_allow_html=True,
    )
    fig = draw_body_silhouettes(light_mode=st.session_state.get("theme") == "light")
    st.pyplot(fig)
    choice = st.selectbox("Select your body type", BODY_TYPES)

    def _go_step3(_choice):
        st.session_state.body_type = _choice
        st.session_state.step      = 3

    st.button("Analyse →", on_click=_go_step3, args=(choice,))


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Photo Capture  (JS component handles live feed + auto-capture)
# ═══════════════════════════════════════════════════════════════════════════════

def _step_capture():
    captured  = st.session_state.photo_captures
    done      = [k for k, _ in VIEWS if k in captured]
    remaining = [(k, lbl) for k, lbl in VIEWS if k not in captured]

    st.markdown(
        "<h1 style='font-size:2rem;font-weight:800;letter-spacing:-0.02em;margin-bottom:0.2rem'>"
        "Step 1 — Capture Photo</h1>",
        unsafe_allow_html=True,
    )
    st.progress(len(done) / len(VIEWS), text=f"{len(done)}/{len(VIEWS)} photos taken")

    if not remaining:
        st.session_state.step = 2
        st.rerun()
        return

    view_key, view_label = remaining[0]
    st.markdown(f"### Shot {len(done)+1}/{len(VIEWS)}")

    guide_col, cam_col = st.columns([1, 2])
    with guide_col:
        st.pyplot(_draw_position_guide(view_key, light_mode=st.session_state.get("theme") == "light"), clear_figure=True)
        if view_key == "face":
            st.caption(
                "📸 **Face close-up.** Used for colour & facial feature analysis.  "
                "Stand ~0.5 m from camera. Fill the frame chin-to-forehead.  "
                "No sunglasses, eyes open, look straight ahead."
            )
        else:
            st.caption(
                "📸 **Full body shot.** Used for body analysis + virtual try-on.  "
                "Stand ~1.5 m back so your **full body is in frame**, "
                "head inside the oval."
            )
    with cam_col:
        if view_key == "face":
            _view_hint = "\ud83d\udcf8 Face close-up \u00b7 fill frame chin to forehead \u00b7 look straight ahead \u00b7 no sunglasses"
        else:
            _view_hint = "\ud83d\udcf8 Full body shot \u00b7 stand ~1.5 m back \u00b7 whole body in frame \u00b7 head inside oval"
        uploaded = _timer_capture_component(view_key, seconds=VIEW_TIMER.get(view_key, 10), hint=_view_hint)

    if uploaded is not None:
        img_bytes = uploaded.read()
        nparr     = np.frombuffer(img_bytes, np.uint8)
        img_bgr   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            st.error("Could not decode image. Please try again.")
            return
        uid      = st.session_state.user_id
        save_dir = CAPTURES_DIR / uid
        save_dir.mkdir(parents=True, exist_ok=True)
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        fpath = save_dir / f"{view_key}_{ts}.jpg"
        cv2.imwrite(str(fpath), img_bgr)
        st.session_state.photo_captures[view_key] = img_bytes
        st.rerun()

    if done:
        st.divider()
        st.markdown("**Captured so far:**")
        cols = st.columns(len(done))
        for col, k in zip(cols, done):
            lbl = next(l for vk, l in VIEWS if vk == k)
            px  = np.frombuffer(captured[k], np.uint8)
            img = cv2.imdecode(px, cv2.IMREAD_COLOR)
            ann = _annotate_thumbnail(img, k)
            col.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), caption=lbl,
                      use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Analysis (auto-runs)
# ═══════════════════════════════════════════════════════════════════════════════

def _step_analysis():
    if not st.session_state.get("body_type"):
        st.warning("Body type is required before analysis.")

        def _go_to_body_type_from_analysis_guard():
            st.session_state.step = 2

        st.button("← Go to Body Type", on_click=_go_to_body_type_from_analysis_guard)
        return

    st.markdown(
        "<h1 style='font-size:2rem;font-weight:800;letter-spacing:-0.02em;margin-bottom:0.5rem'>"
        "Step 3 — Analysing</h1>",
        unsafe_allow_html=True,
    )
    face_lmk, pose_lmk, segmenter = _load_models_image()
    captured = st.session_state.photo_captures
    results  = {}
    pbar     = st.progress(0)

    processed = 0
    total_views = sum(1 for k, _ in VIEWS if k in captured)
    for view_key, view_label in VIEWS:
        if view_key not in captured:
            continue
        st.markdown(f"Processing **{view_label}**…")
        px    = np.frombuffer(captured[view_key], np.uint8)
        img   = cv2.imdecode(px, cv2.IMREAD_COLOR)
        try:
            attrs = analyze_view(img, view_key, face_lmk, pose_lmk, segmenter)
        except Exception as _exc:
            st.error(f"Analysis failed for **{view_label}**: {_exc}")

            def _retake_photos_from_error():
                st.session_state.photo_captures = {}
                st.session_state.step = 1

            st.button("← Retake Photos", on_click=_retake_photos_from_error)
            return
        results[view_key] = attrs
        processed += 1
        pbar.progress(processed / max(total_views, 1))

    st.session_state.view_results = results
    st.session_state.final_attrs  = fuse_results(results, st.session_state.body_type)

    # Auto-save attributes to profile under the user's UUID
    uid = st.session_state.get("user_id")
    if uid:
        sid = get_repository().add_analysis_session(
            uid,
            st.session_state.final_attrs,
            list(captured.keys()),
        )
        if sid:
            st.session_state.current_session_id     = sid
            st.session_state.current_rec_session_id = None  # reset for new analysis

    st.session_state.step = 5
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Results
# ═══════════════════════════════════════════════════════════════════════════════

def _step_results():
    st.markdown(
        "<h1 style='font-size:2rem;font-weight:800;letter-spacing:-0.02em;margin-bottom:0.5rem'>"
        "✦ Your FITS Analysis</h1>",
        unsafe_allow_html=True,
    )
    attrs = st.session_state.final_attrs
    if not attrs:
        st.warning("No results yet.")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Colours")
        _swatch_row_main("Skin",  attrs.get("skin_color"))
        _swatch_row_main("Eyes",  attrs.get("eyes_color"))
        _swatch_row_main("Lips",  attrs.get("lips_color"))
        _swatch_row_main("Hair",  attrs.get("hair_color"))
        st.markdown("### Attributes")
        st.markdown(f"**Face Shape**: {attrs.get('face_shape') or '—'}")
        st.markdown(f"**Body Type**: {attrs.get('body_type') or '—'}")
        st.markdown(f"**Hair Length**: {attrs.get('hair_length') or '—'}")
        st.markdown(f"**Neck Length**: {attrs.get('neck_length') or '—'}")
        st.markdown(f"**Undertone**: {attrs.get('undertone') or '—'}")

    with col2:
        st.markdown("### Body Silhouette")
        fig = draw_body_silhouettes(highlight=attrs.get("body_type"), light_mode=st.session_state.get("theme") == "light")
        st.pyplot(fig)

    captured = st.session_state.get("photo_captures", {})
    if captured:
        with st.expander("🎯 Facial Landmarks", expanded=False):
            face_lmk, pose_lmk, seg = _load_models_image()
            lm_cols = st.columns(len(captured))
            for lcol, (vkey, raw_bytes) in zip(lm_cols, captured.items()):
                nparr   = np.frombuffer(raw_bytes, np.uint8)
                img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img_bgr is not None:
                    ann = draw_face_landmarks(img_bgr, face_lmk, pose_lmk, seg)
                    lcol.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB),
                               caption=vkey, use_container_width=True)

    view_results = st.session_state.get("view_results", {})
    if view_results:
        with st.expander("📷 Per-view Breakdown", expanded=False):
            COLOUR_KEYS = ["skin_color","eyes_color","lips_color","hair_color"]
            TEXT_KEYS   = ["face_shape","hair_length","neck_length","undertone"]
            vcols = st.columns(len(view_results))
            for vcol, (vname, vr) in zip(vcols, view_results.items()):
                vcol.markdown(f"**{vname.replace('_',' ').title()}**")
                for ck in COLOUR_KEYS:
                    h = vr.get(ck)
                    if h and h != "#000000":
                        vcol.markdown(
                            f'<div style="display:flex;align-items:center;gap:6px">'
                            f'<div style="width:18px;height:18px;border-radius:3px;'
                            f'background:{h};border:1px solid {_theme_color("#e0d9f7", "#555", "#58554c")}"></div>'
                            f'<span style="font-size:11px">{ck.replace("_color","")}: {h}</span></div>',
                            unsafe_allow_html=True)
                for tk in TEXT_KEYS:
                    val = vr.get(tk)
                    if val:
                        vcol.markdown(f"<span style='font-size:11px'>{tk}: {val}</span>",
                                      unsafe_allow_html=True)

    st.divider()
    if st.session_state.get("current_session_id"):
        st.caption(f"✅ Profile saved · session `{st.session_state.current_session_id[:8]}…`")
    c_dl, c_rec, c_over = st.columns(3)
    with c_dl:
        st.download_button(
            "⬇ Download JSON",
            data=json.dumps(attrs, indent=2),
            file_name="fits_attributes.json",
            mime="application/json",
            use_container_width=True,
        )
    with c_rec:

        def _go_recommendations_from_results():
            # Reset recommender state for a fresh run
            st.session_state.rec_results    = None
            st.session_state.rec_query      = ""
            st.session_state.image_index    = {}
            st.session_state.step           = 5

        st.button("✨ Get Recommendations →", use_container_width=True, type="primary", on_click=_go_recommendations_from_results)
    with c_over:

        def _restart_from_results():
            for k in list(st.session_state.keys()):
                del st.session_state[k]

        st.button("🔄 Start Over", use_container_width=True, on_click=_restart_from_results)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Recommendations (Task 2 integrated)
# ═══════════════════════════════════════════════════════════════════════════════

def _quiz_palette_to_tone(palette: str) -> str:
    """Map quiz colour palette answer to recommender skin_tone value."""
    p = (palette or "").lower()
    if "neutral" in p or "dark" in p or "pastel" in p:
        return "cool"
    if "earth" in p or "bold" in p:
        return "warm"
    return ""


def _render_style_quiz(user_name: str):
    """
    Fashion personality quiz — feels fun & like it tunes recommendations,
    while collecting study-relevant data about shopping behaviour and style attitude.
    Answers are saved persistently per user and passed to the experiment record.
    """
    _sub_q = _theme_color("#7c6fa0", "#8b949e", "#b3ad9f")
    st.markdown(
        f"<h1 style='font-size:2rem;font-weight:800;letter-spacing:-0.02em;margin-bottom:0.3rem'>"
        f"✦ Quick Style Check</h1>"
        f"<p style='color:{_sub_q};margin-bottom:1.4rem'>"
        f"5 quick questions so we can personalise your picks. Totally optional — "
        f"hit <b>Skip</b> to jump straight to recommendations.</p>",
        unsafe_allow_html=True,
    )

    with st.form("style_quiz_form"):
        st.markdown("**1. What's your go-to style vibe?**")
        vibe = st.radio(
            "vibe",
            ["Streetwear / Urban", "Clean & Minimal", "Preppy / Smart-casual",
             "Vintage / Retro", "Boho / Relaxed", "I mix it all up"],
            index=5, horizontal=True, label_visibility="collapsed",
        )

        st.markdown("**2. When you're shopping, you're usually…**")
        shopping_mode = st.radio(
            "shopping_mode",
            ["Looking for something specific", "Browsing for inspiration",
             "Trying something new / out of my comfort zone", "Building a complete outfit"],
            index=1, horizontal=True, label_visibility="collapsed",
        )

        st.markdown("**3. How would you describe your approach to fashion?**")
        adventurousness = st.select_slider(
            "adventurousness",
            options=["Play it safe — I stick to what works",
                     "Mostly safe, occasional experiment",
                     "Equal mix of safe and bold",
                     "Mostly adventurous, sometimes classic",
                     "Always experimenting with new looks"],
            value="Equal mix of safe and bold",
            label_visibility="collapsed",
        )

        st.markdown("**4. What matters MOST to you in an outfit?**")
        priority = st.radio(
            "priority",
            ["Comfort", "Style & aesthetics", "Versatility (wears many ways)",
             "Brand / quality", "Price / value"],
            index=0, horizontal=True, label_visibility="collapsed",
        )

        st.markdown("**5. How often do you buy new clothes?**")
        buy_freq = st.radio(
            "buy_freq",
            ["Rarely — only when I need to", "A few times a year",
             "Monthly", "Weekly or more"],
            index=1, horizontal=True, label_visibility="collapsed",
        )

        save_btn = st.form_submit_button("Let's go →", use_container_width=True, type="primary")

    _, skip_col = st.columns([3, 1])
    with skip_col:

        def _skip_style_quiz():
            st.session_state.style_quiz = {}
            st.session_state["_force_quiz_edit"] = False

        st.button("Skip →", use_container_width=True, on_click=_skip_style_quiz)

    if save_btn:
        st.session_state.style_quiz = {
            "vibe":            vibe,
            "shopping_mode":   shopping_mode,
            "adventurousness": adventurousness,
            "priority":        priority,
            "buy_frequency":   buy_freq,
        }
        uid = st.session_state.get("user_id")
        st.session_state["_force_quiz_edit"] = False
        if uid:
            repo = get_repository()
            saved_prefs = repo.get_user_preferences(uid) or {}
            saved_prefs["style_quiz"] = st.session_state.style_quiz
            repo.save_user_preferences(uid, saved_prefs)
        st.rerun()


def _step_recommendations():
    user_name = st.session_state.get("user_name", "You")

    # ── Load saved preferences if not already in session ───────────────────────
    if st.session_state.get("style_quiz") is None and not st.session_state.get("_force_quiz_edit"):
        uid = st.session_state.get("user_id")
        if uid:
            repo = get_repository()
            saved_prefs = repo.get_user_preferences(uid) or {}
            saved_quiz = saved_prefs.get("style_quiz")
            saved_filters = saved_prefs.get("sidebar_filters")
            
            # Restore quiz answers if previously saved
            if saved_quiz:
                st.session_state.style_quiz = saved_quiz
            # Restore sidebar filters if previously saved
            if saved_filters:
                st.session_state["_last_occasion"] = saved_filters.get("occasion", "")
                st.session_state["_last_clothing"] = saved_filters.get("clothing_type", "")
                st.session_state["_last_gender"] = saved_filters.get("gender", "")
                st.session_state["_last_skin_tone"] = saved_filters.get("skin_tone", "")
                st.session_state["_last_body_type"] = saved_filters.get("body_type", "")
                st.session_state["_last_face_shape"] = saved_filters.get("face_shape", "")

    # ── Restore last rec results + experiment if session was cleared ───────────
    if st.session_state.get("rec_results") is None and not st.session_state.get("_force_quiz_edit"):
        uid = st.session_state.get("user_id")
        if uid:
            exp_repo = get_experiment_repository()
            _last_exp = getattr(exp_repo, 'get_latest_experiment', lambda _: None)(uid)
            if _last_exp:
                # Enrich saved item records with images from current inventory
                _inv_map = {
                    str(it.get("id", "")): it
                    for it in _recommender().inventory
                }
                def _enrich(record: dict) -> dict:
                    """Merge saved experiment record with live inventory data for images."""
                    inv_item = _inv_map.get(str(record.get("id", "")), {})
                    return {
                        **inv_item,                          # images, price, etc. from inventory
                        "id":       record["id"],
                        "name":     record.get("name", inv_item.get("name", "")),
                        "brand":    record.get("brand", inv_item.get("brand", "")),
                        "category": record.get("category", inv_item.get("category", "")),
                        "score":    record.get("score", 0),
                        "adjusted_score": record.get("score", 0),
                        "source":   record.get("source", "ranked"),
                        "color_tone":    record.get("metadata", {}).get("color_tone", ""),
                        "sleeve_length": record.get("metadata", {}).get("sleeve_length", ""),
                        "feedback": record.get("feedback"),
                    }
                _restored = (
                    [_enrich(r) for r in _last_exp.get("ranked_items", [])]
                    + [_enrich(r) for r in _last_exp.get("discovery_items", [])]
                )
                if _restored:
                    st.session_state.rec_results = _restored
                    st.session_state.current_experiment_id = _last_exp["exp_id"]
                    # Restore per-product feedback into session state
                    for r in _last_exp.get("ranked_items", []) + _last_exp.get("discovery_items", []):
                        if r.get("feedback"):
                            st.session_state.feedback[r["id"]] = r["feedback"]
                    # Restore filters from experiment
                    _ef = _last_exp.get("filters", {})
                    if _ef.get("occasion"):
                        st.session_state["_last_occasion"] = _ef["occasion"]
                    if _ef.get("clothing_type"):
                        st.session_state["_last_clothing"] = _ef["clothing_type"]
                    if _ef.get("gender"):
                        st.session_state["_last_gender"] = _ef["gender"]

    # ── Style quiz gate — shown once if not yet taken ─────────────────────────
    if st.session_state.get("style_quiz") is None:
        _render_style_quiz(user_name)
        return

    _hero_name_col = _theme_color("#1a1033", "#e2e8f0", "#ece8dd")
    st.markdown(
        f"<div class='rec-hero'>"
        f"<div><p class='rec-hero-title'>✦ FITS Recommendations</p>"
        f"<p class='rec-hero-sub'>Personalised picks for <b style='color:{_hero_name_col}'>{user_name}</b></p></div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <style>
        .rec-filter-attn {
            border: 2px solid #f59e0b;
            background: linear-gradient(135deg, rgba(245,158,11,0.16), rgba(245,158,11,0.08));
            color: inherit;
            border-radius: 14px;
            padding: 12px 14px;
            margin: 0.4rem 0 0.8rem 0;
            animation: recPulse 1.4s ease-in-out infinite;
        }
        .rec-filter-attn strong {
            font-size: 1.02rem;
        }
        @keyframes recPulse {
            0%, 100% { box-shadow: 0 0 0 0 rgba(245,158,11,0.35); }
            50% { box-shadow: 0 0 0 8px rgba(245,158,11,0.05); }
        }
        </style>
        <div class='rec-filter-attn'>
            <strong>⚠️ Action needed:</strong> complete the <strong>sidebar filters</strong> first,
            then click <strong>Get Recommendations</strong>.
            Required: <strong>Occasion</strong> and <strong>Clothing Type</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.warning("👈 Use the sidebar on the left to set filters before requesting recommendations.")

    _uid_for_toast = st.session_state.get("user_id")
    _show_rec_toast = False
    if _uid_for_toast:
        _repo_for_toast = get_repository()
        _prefs_for_toast = _repo_for_toast.get_user_preferences(_uid_for_toast) or {}
        if not _prefs_for_toast.get("rec_filters_toast_shown", False):
            _show_rec_toast = True
            _prefs_for_toast["rec_filters_toast_shown"] = True
            _repo_for_toast.save_user_preferences(_uid_for_toast, _prefs_for_toast)
    else:
        # Fallback for anonymous sessions with no user_id.
        if not st.session_state.get("_rec_filters_toast_shown", False):
            _show_rec_toast = True
            st.session_state["_rec_filters_toast_shown"] = True

    if _show_rec_toast:
        st.toast("Set sidebar filters first (Occasion + Clothing Type), then tap Get Recommendations.", icon="⚠️")

    attrs = st.session_state.final_attrs or {}

    # ── Bridge Task 1 attrs to Task 2 defaults ────────────────────────────────
    default_face_shape = (attrs.get("face_shape") or "").lower()
    default_body_type  = _body_type_to_t2(attrs.get("body_type") or "")
    default_skin_tone  = _undertone_to_skin_tone(attrs.get("undertone") or "")
    default_neck       = _neck_length_to_t2(attrs.get("neck_length") or "")
    default_hair_cat   = _hex_to_hair_category(attrs.get("hair_color") or "")

    # ── Sidebar filter panel ──────────────────────────────────────────────────
    with st.sidebar:
        st.divider()
        _sb_lbl  = _theme_color("#7c6fa0", "#8b949e", "#b8b09b")
        _sb_sub  = _theme_color("#9c88cc", "#6e7681", "#a39f93")
        st.markdown(
            f"<p style='font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;color:{_sb_lbl};font-weight:700;margin-bottom:4px'>🎛️ Recommendation Filters</p>"
            f"<p style='font-size:0.78rem;color:{_sb_sub};margin-bottom:8px'>Auto-filled from analysis · adjust as needed</p>",
            unsafe_allow_html=True,
        )

        with st.form("rec_form"):
            gender = st.selectbox("Gender", ["", "women", "men"], index=0)

            face_shape_opts = ["", "round", "square", "oval", "heart", "oblong"]
            fs_idx = face_shape_opts.index(default_face_shape) if default_face_shape in face_shape_opts else 0
            face_shape = st.selectbox("Face Shape", face_shape_opts, index=fs_idx)

            body_opts = ["", "pear", "rectangle", "hourglass", "inverted_triangle"]
            bt_idx = body_opts.index(default_body_type) if default_body_type in body_opts else 0
            body_type = st.selectbox("Body Type", body_opts, index=bt_idx)

            skin_opts = ["", "cool", "warm"]
            sk_idx = skin_opts.index(default_skin_tone) if default_skin_tone in skin_opts else 0
            skin_tone = st.selectbox("Skin Tone", skin_opts, index=sk_idx)

            neck_opts = ["", "short", "average", "long"]
            nk_idx = neck_opts.index(default_neck) if default_neck in neck_opts else 0
            neck_length = st.selectbox("Neck Length", neck_opts, index=nk_idx)

            hair_opts = ["", "blonde", "brunette", "black", "red", "gray"]
            hc_idx = hair_opts.index(default_hair_cat) if default_hair_cat in hair_opts else 0
            hair_color = st.selectbox("Hair Color", hair_opts, index=hc_idx)

            height_opts = ["", "petite", "average", "tall"]
            height = st.selectbox("Height", height_opts, index=0)

            weight_opts = ["", "slim", "mid", "full"]
            weight_range = st.selectbox("Weight Range", weight_opts, index=0)

            occasion = st.selectbox("Occasion ✦", ["", "sporty", "casual", "event", "formals"])
            clothing_type = st.selectbox(
                "Clothing Type ✦",
                ["", "dress", "top", "shorts", "trousers", "skirt"],
            )

            # Brand filter — all brands selected by default, hidden until expanded
            _all_brands = sorted({
                str(it.get("brand", ""))
                for it in _recommender().inventory
                if it.get("brand")
            })
            with st.expander("🏷️ Brand Filter", expanded=False):
                selected_brands = st.multiselect(
                    "Brands",
                    options=_all_brands,
                    default=_all_brands,
                    help="Deselect brands you want to exclude",
                    label_visibility="collapsed",
                )

            submitted = st.form_submit_button("✨ Get Recommendations", use_container_width=True)

        sq_snap = st.session_state.get("style_quiz") or {}
        if sq_snap:
            _vibe_label = sq_snap.get("vibe", "")
            if _vibe_label:
                st.caption(f"✓ Vibe: {_vibe_label}")

        def _edit_style_quiz():
            st.session_state.style_quiz = None
            st.session_state["_force_quiz_edit"] = True

        st.button("✏️ Edit Style Quiz", use_container_width=True, on_click=_edit_style_quiz)

    # ── Profile summary pills ─────────────────────────────────────────────────
    if attrs:
        skin_hex = attrs.get("skin_color", "")
        pills_html = ""
        for label, key in [("Face", "face_shape"),("Body", "body_type"),("Tone", "undertone"),("Neck", "neck_length")]:
            val = attrs.get(key) or ""
            if val:
                pills_html += f"<span class='filter-pill'>{label}: {val}</span>"
        if skin_hex:
            pills_html += (
                f"<span class='filter-pill' style='background:#f3e8ff;color:#7c3aed;border-color:#e9d5ff'>"
                f"<span style='display:inline-block;width:10px;height:10px;border-radius:50%;"
                f"background:{skin_hex};margin-right:4px;border:1px solid #ccc'></span>Skin</span>"
            )
        if pills_html:
            st.markdown(f"<div style='margin-bottom:0.8rem'>{pills_html}</div>", unsafe_allow_html=True)

    st.divider()

    # ── Run recommendations when form submitted ───────────────────────────────
    if submitted:
        # lifestyle from quiz can satisfy the occasion requirement as a fallback
        _sq_now = st.session_state.get("style_quiz") or {}
        _LIFESTYLE_OCC = {
            "Work":      "formals",
            "Going-Out": "event",
            "Active":    "sporty",
            "Formal":    "formals",
        }
        _effective_occasion = occasion or _LIFESTYLE_OCC.get(_sq_now.get("lifestyle", ""), "")
        if not _effective_occasion or not clothing_type:
            st.error(
                "⚠️ Please select **Clothing Type ✦** in the sidebar"
                + (" and an **Occasion** (or set your lifestyle in the Style Quiz to Work / Active / Going-Out / Formal)" if not _effective_occasion else "")
                + " before getting recommendations."
            )
        else:
            user_attrs = {
                "gender":       gender,
                "face_shape":   face_shape,
                "body_type":    body_type,
                "skin_tone":    skin_tone,
                "height":       height,
                "weight_range": weight_range,
                "neck_length":  neck_length,
                "hair_color":   hair_color,
                "occasion":     _effective_occasion,
                "clothing_type": clothing_type,
                # Only pass brand filter if user deselected some brands
                "_brand_filter": selected_brands if len(selected_brands) < len(_all_brands) else None,
            }
            
            # Apply quiz vibe as a light query nudge
            sq = st.session_state.get("style_quiz") or {}
            _VIBE_MAP = {
                "Streetwear / Urban":       "streetwear urban casual graphic logo",
                "Clean & Minimal":          "minimal clean solid simple classic",
                "Preppy / Smart-casual":    "preppy smart-casual polo button-up tailored",
                "Vintage / Retro":          "vintage retro classic heritage washed",
                "Boho / Relaxed":           "boho relaxed flowy oversized earthy",
            }
            _vibe = sq.get("vibe", "")
            if _vibe and _vibe in _VIBE_MAP:
                user_attrs["_vibe_hint"] = _VIBE_MAP[_vibe]
            
            # Persist filter snapshot so the save button can read them
            st.session_state["_last_occasion"]   = occasion
            st.session_state["_last_clothing"]   = clothing_type
            st.session_state["_last_gender"]     = gender
            st.session_state["_last_skin_tone"]  = skin_tone
            st.session_state["_last_body_type"]  = body_type
            st.session_state["_last_face_shape"] = face_shape
            with st.spinner("Finding your best matches…"):
                engine = _recommender()
                # Fresh UUID seed each submit → genuinely new discovery picks every session
                # for data-collection purposes (feedback on discovery items tunes fashion_rules)
                _rec_seed = hash(
                    (st.session_state.get("user_id") or "anon") + str(uuid.uuid4())
                )
                _rec_out = engine.recommend(user_attrs, clothing_type, random_seed=_rec_seed)
                if isinstance(_rec_out, tuple) and len(_rec_out) == 3:
                    query_text, results, filter_trace = _rec_out
                else:
                    query_text, results = _rec_out
                    filter_trace = []
            if results:
                results = apply_feedback_rerank(results)
                st.session_state.rec_results       = results
                st.session_state.rec_query         = query_text
                st.session_state.rec_filter_trace  = filter_trace
                st.session_state.image_index       = {}
            else:
                st.session_state.rec_results      = []
                st.session_state.rec_query        = query_text
                st.session_state.rec_filter_trace = filter_trace

            # Auto-save recommendation session nested under the user's UUID
            uid = st.session_state.get("user_id")
            if uid and st.session_state.rec_results:
                _autosave_rec_session(uid, st.session_state.rec_results)
                
                # Also save sidebar filters persistently
                repo = get_repository()
                saved_prefs = repo.get_user_preferences(uid) or {}
                saved_prefs["sidebar_filters"] = {
                    "occasion":     st.session_state.get("_last_occasion", ""),
                    "clothing_type": st.session_state.get("_last_clothing", ""),
                    "gender":       st.session_state.get("_last_gender", ""),
                    "skin_tone":    st.session_state.get("_last_skin_tone", ""),
                    "body_type":    st.session_state.get("_last_body_type", ""),
                    "face_shape":   st.session_state.get("_last_face_shape", ""),
                }
                repo.save_user_preferences(uid, saved_prefs)

    # ── Display results ───────────────────────────────────────────────────────
    if st.session_state.rec_results is None:
        st.error(
            "No recommendations yet. Please fill the sidebar filters first "
            "(required: **Occasion** and **Clothing Type**) and click **Get Recommendations**."
        )
        return

    results = st.session_state.rec_results
    if not results:
        filter_trace = st.session_state.get("rec_filter_trace") or []
        # Find the first filter that reduced the pool to zero
        eliminator = next((f for f in filter_trace if f["after"] == 0), None)
        if eliminator:
            st.warning(
                f"No products found — the **{eliminator['name']}** filter eliminated all candidates "
                f"({eliminator['before']} → {eliminator['after']})."
            )
        else:
            st.warning("No matching products found. Try broadening your filters.")
        if filter_trace:
            with st.expander("Filter breakdown"):
                for step in filter_trace:
                    arrow = "🚫" if step["after"] == 0 else ("⚠️" if step["after"] < 5 else "✓")
                    st.caption(f"{arrow} **{step['name']}**: {step['before']} → {step['after']} items")
        return

    display_results = results

    ranked_results    = [r for r in display_results if r.get("source", "ranked") == "ranked"]
    discovery_results = [r for r in display_results if r.get("source") == "random"]

    # ── Active filter pills ───────────────────────────────────────────────────
    active_pill_parts = []
    for _key, _label in [("_last_occasion","Occasion"),("_last_clothing","Type"),("_last_gender","Gender")]:
        _v = st.session_state.get(_key,"")
        if _v:
            active_pill_parts.append(f"<span class='filter-pill'>{_label}: {_v}</span>")
    if st.session_state.get("_last_skin_tone"):
        active_pill_parts.append(f"<span class='filter-pill'>Tone: {st.session_state['_last_skin_tone']}</span>")
    pills_row   = " ".join(active_pill_parts)
    count_badge = f"<span class='count-badge'>{len(ranked_results) + len(discovery_results)} picks</span>"
    st.markdown(
        f"<div style='display:flex;align-items:center;justify-content:space-between;"
        f"flex-wrap:wrap;gap:6px;margin-bottom:0.8rem'>"
        f"<div>{pills_row}</div>{count_badge}</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"Query: _{st.session_state.rec_query}_")

    # ── Shared card renderer ──────────────────────────────────────────────────
    def _draw_card(col, item, key_sfx):
        """Render a single product card inside *col*."""
        with col:
            product_id = item.get("id", key_sfx)
            images     = item.get("images", [])
            state_key  = f"img_{product_id}"
            if state_key not in st.session_state.image_index:
                st.session_state.image_index[state_key] = 0
            is_sporty = str(st.session_state.get("_last_occasion","")).lower() == "sporty"

            st.markdown('<div class="product-card">', unsafe_allow_html=True)

            # Image carousel
            if images:
                _idx = st.session_state.image_index[state_key]
                _idx = max(0, min(_idx, len(images) - 1))
                try:
                    st.image(images[_idx], use_container_width=True)
                except Exception:
                    st.info("📸 Image unavailable")
                if len(images) > 1:
                    def _prev_img(_sk=state_key, _imgs=images):
                        st.session_state.image_index[_sk] = (st.session_state.image_index.get(_sk, 0) - 1) % len(_imgs)
                    def _next_img(_sk=state_key, _imgs=images):
                        st.session_state.image_index[_sk] = (st.session_state.image_index.get(_sk, 0) + 1) % len(_imgs)
                    _b1, _, _b3 = st.columns([1, 1, 1])
                    with _b1:
                        st.button("⬅️", key=f"left_{key_sfx}", use_container_width=True, on_click=_prev_img)
                    with _b3:
                        st.button("➡️", key=f"right_{key_sfx}", use_container_width=True, on_click=_next_img)
            else:
                st.info("📸 No image available")

            # Brand + title
            _brand = item.get("brand", "")
            _title = item.get("name", "")
            _sporty_badge = "<span class='badge-sporty'>🏃 Sport</span>" if is_sporty else ""
            _brand_html   = f"<p class='product-brand'>{_brand}{_sporty_badge}</p>" if _brand else ""
            st.markdown(
                f"<div class='card-body'>{_brand_html}<p class='product-title'>{_title}</p></div>",
                unsafe_allow_html=True,
            )

            # Try-On button
            if images:
                _shown = images[st.session_state.image_index.get(state_key, 0)]
                def _select_for_tryon(_item=item, _shown_img=_shown):
                    _comp  = _infer_component(_item)
                    _locks = st.session_state.get("outfit_locks", {})
                    _sel   = st.session_state.get("outfit_components", {})
                    _prior = _sel.get(_comp) if _comp else None
                    if _comp and _locks.get(_comp) and _prior and str(_prior.get("id")) != str(_item.get("id")):
                        st.session_state["_tryon_lock_warning"] = {
                            "item_id": str(_item.get("id", "")),
                            "msg": f"{_comp.title()} is locked. Unlock it before replacing.",
                        }
                        return
                    st.session_state.pop("_tryon_lock_warning", None)
                    st.session_state.tryon_selection_request = {
                        "item": _item,
                        "shown_image": _shown_img,
                        "component": _comp,
                    }

                st.button(
                    "👗 Try On",
                    key=f"tryon_{key_sfx}",
                    use_container_width=True,
                    disabled=st.session_state.tryon_count <= 0,
                    on_click=_select_for_tryon,
                )
                _tryon_warn = st.session_state.get("_tryon_lock_warning")
                if _tryon_warn and _tryon_warn.get("item_id") == str(item.get("id", "")):
                    st.warning(_tryon_warn.get("msg", "This garment is locked."))

            # Feedback buttons
            def _set_feedback(_value, _pid=product_id):
                st.session_state.feedback[_pid] = _value
                _uid = st.session_state.get("user_id")
                if _uid:
                    _autoupdate_feedback(_uid)
                # Track in experiment
                exp_repo = get_experiment_repository()
                exp_repo.update_feedback(
                    st.session_state.get("current_experiment_id"),
                    _pid,
                    _value
                )
                if st.session_state.rec_results:
                    st.session_state.rec_results = apply_feedback_rerank(
                        st.session_state.rec_results
                    )

            _c1, _c2, _c3, _c4 = st.columns([2, 2, 1, 1])
            with _c1:
                st.button(
                    "👍",
                    key=f"like_{key_sfx}",
                    use_container_width=True,
                    help="Like",
                    on_click=_set_feedback,
                    args=("like",),
                )
            with _c2:
                st.button(
                    "👎",
                    key=f"dislike_{key_sfx}",
                    use_container_width=True,
                    help="Dislike",
                    on_click=_set_feedback,
                    args=("dislike",),
                )
            with _c3:
                _cur_fb = st.session_state.feedback.get(product_id)
                if _cur_fb:
                    _fb_icon = "❤️" if _cur_fb == "like" else "💔"
                    st.markdown(
                        f"<div style='text-align:center;padding-top:6px'>{_fb_icon}</div>",
                        unsafe_allow_html=True,
                    )
            with _c4:
                _sc_key = f"show_score_{product_id}"
                if _sc_key not in st.session_state:
                    st.session_state[_sc_key] = False

                def _toggle_score(_sk=_sc_key):
                    st.session_state[_sk] = not st.session_state.get(_sk, False)

                st.button(
                    "📊",
                    key=f"scbtn_{key_sfx}",
                    use_container_width=True,
                    help="Dev: show score",
                    on_click=_toggle_score,
                )
            if st.session_state.get(_sc_key):
                _sc_col = _theme_color("#7c6fa0", "#8b949e", "#a9a497")
                st.markdown(
                    f"<p style='font-size:0.7rem;color:{_sc_col};margin:2px 0 0;text-align:right'>"
                    f"score: {round(item.get('adjusted_score', item.get('score', 0)), 4)}</p>",
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)

    # ── Section 1: Ranked Picks ───────────────────────────────────────────────
    _rp_accent = _theme_color("#16a34a", "#3fb950", "#9fb89f")
    _rp_sub    = _theme_color("#7c6fa0", "#8b949e", "#a9a497")
    st.markdown(
        f"<h3 style='margin:1.2rem 0 0.4rem;font-size:1.1rem;color:{_rp_accent}'>"
        f"✦ Ranked Picks</h3>"
        f"<p style='font-size:0.8rem;color:{_rp_sub};margin-bottom:0.8rem'>"
        f"Top 10 matches based on your style profile and attributes.</p>",
        unsafe_allow_html=True,
    )
    if ranked_results:
        for _row_start in range(0, len(ranked_results), 3):
            _row_items = ranked_results[_row_start:_row_start + 3]
            _rcols = st.columns(3)
            for _ri_offset, _ritem in enumerate(_row_items):
                _rid = str(_ritem.get("id", f"{_row_start + _ri_offset}"))
                _draw_card(_rcols[_ri_offset], _ritem, f"r_{_rid}")
    else:
        st.info("No ranked picks available.")

    _hr_col = _theme_color("#e0d9f7", "#30363d", "#3a3d43")
    st.markdown(f"<hr style='border-color:{_hr_col};margin:1.8rem 0'>", unsafe_allow_html=True)

    # ── Section 2: Discovery Picks ────────────────────────────────────────────
    _dp_accent = _theme_color("#7c3aed", "#c084fc", "#b8b09b")
    _dp_sub    = _theme_color("#7c6fa0", "#8b949e", "#a9a497")
    st.markdown(
        f"<h3 style='margin:0.4rem 0 0.4rem;font-size:1.1rem;color:{_dp_accent}'>"
        f"🎲 Discovery Picks</h3>"
        f"<p style='font-size:0.8rem;color:{_dp_sub};margin-bottom:0.8rem'>"
        f"10 random selections outside your usual profile — rate these to help tune your fashion rules.</p>",
        unsafe_allow_html=True,
    )
    if discovery_results:
        for _row_start in range(0, len(discovery_results), 3):
            _row_items = discovery_results[_row_start:_row_start + 3]
            _dcols = st.columns(3)
            for _di_offset, _ditem in enumerate(_row_items):
                _did = str(_ditem.get("id", f"{_row_start + _di_offset}"))
                _draw_card(_dcols[_di_offset], _ditem, f"d_{_did}")
    else:
        st.info("No discovery picks available.")

    # ── Auto-save status ───────────────────────────────────────────────────────
    if st.session_state.get("current_rec_session_id") and st.session_state.get("user_id"):
        _all_shown = ranked_results + discovery_results  # noqa: F821
        current_ids    = {item.get("id", str(i)) for i, item in enumerate(_all_shown)}
        liked_count    = sum(1 for pid, v in st.session_state.feedback.items() if v == "like"    and pid in current_ids)
        disliked_count = sum(1 for pid, v in st.session_state.feedback.items() if v == "dislike" and pid in current_ids)
        rated_count    = liked_count + disliked_count
        total_count    = len(_all_shown)
        sid_short = st.session_state.current_rec_session_id[:8]
        fb_parts = []
        if liked_count:
            fb_parts.append(f"{liked_count} liked ❤️")
        if disliked_count:
            fb_parts.append(f"{disliked_count} disliked 💔")
        rated_str = f"{rated_count}/{total_count} rated"
        fb_str = " · " + " · ".join([rated_str] + fb_parts) if fb_parts else f" · {rated_str}"
        _as_light = st.session_state.get("theme") == "light"
        _as_col  = "#16a34a" if _as_light else "#3fb950"
        _as_code = "background:#ede9fe" if _as_light else "background:#161b22"
        st.markdown(
            f"<p style='font-size:0.75rem;color:{_as_col};margin-top:1rem'>"
            f"✅ Auto-saved &nbsp;·&nbsp; <code style='{_as_code};padding:1px 5px;border-radius:4px'>{sid_short}…</code>{fb_str}</p>",
            unsafe_allow_html=True,
        )

    # Process pending try-on request after the card grid has fully rendered.
    _tryon_req = st.session_state.get("tryon_selection_request")
    if _tryon_req:
        _item = _tryon_req.get("item") or {}
        _shown_img = _tryon_req.get("shown_image")
        _comp = _tryon_req.get("component")
        _sel = st.session_state.get("outfit_components", {})
        if _comp:
            _sel[_comp] = {
                "id":       _item.get("id", ""),
                "name":     _item.get("name", ""),
                "brand":    _item.get("brand", ""),
                "category": _item.get("category", ""),
                "image":    _shown_img,
            }
            st.session_state.outfit_components = _sel
            st.session_state.tryon_expected_component = _comp
        st.session_state.tryon_expected_item_id = _item.get("id", "")
        st.session_state.tryon_expected_image   = _shown_img
        st.session_state.tryon_item             = _item
        st.session_state.tryon_image            = _shown_img
        st.session_state.tryon_result           = None
        st.session_state.tryon_selection_request = None

        _exp_id = st.session_state.get("current_experiment_id")
        _item_id = _item.get("id", "")
        if _exp_id and _item_id:
            _tryon_repo = get_experiment_repository()
            _garment_type = st.session_state.get("_selected_garment_type", "full")
            _tryon_repo.record_tryon_selection(_exp_id, _item_id, _garment_type)

        st.session_state.step = 6
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Virtual Try-On
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_upload_image(img_bytes: bytes, label: str, max_side: int = 1280) -> tuple[bytes | None, str | None]:
    """Decode, optionally downscale, and re-encode to JPEG for stable local inference."""
    if not img_bytes:
        return None, f"{label} image payload is empty."
    try:
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None, f"{label} image could not be decoded."

        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            return None, f"{label} image has invalid dimensions."

        longest = max(h, w)
        if longest > max_side:
            scale = max_side / float(longest)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            return None, f"{label} image could not be re-encoded as JPEG."
        return buf.tobytes(), None
    except Exception as exc:
        return None, f"{label} image normalization failed: {type(exc).__name__}: {exc}"


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _write_cloth_mask_jpg(cloth_bgr: np.ndarray, out_path: Path):
    """Create a simple binary cloth mask (white foreground) as HR-VITON cloth-mask input."""
    gray = cv2.cvtColor(cloth_bgr, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    inv = cv2.medianBlur(inv, 5)
    kernel = np.ones((5, 5), np.uint8)
    inv = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel, iterations=2)
    cv2.imwrite(str(out_path), inv)


def _write_openpose_like(person_bgr: np.ndarray, openpose_img_path: Path, openpose_json_path: Path):
    """Generate OpenPose-like files from MediaPipe pose for HR-VITON compatibility."""
    h, w = person_bgr.shape[:2]
    rgb = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB)
    pts = np.zeros((18, 3), dtype=np.float32)

    try:
        with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1) as pose:
            res = pose.process(rgb)
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark

            def _xy(idx):
                return np.array([lm[idx].x * w, lm[idx].y * h], dtype=np.float32), float(lm[idx].visibility)

            mp_idx = mp.solutions.pose.PoseLandmark
            left_sh, lsv = _xy(mp_idx.LEFT_SHOULDER.value)
            right_sh, rsv = _xy(mp_idx.RIGHT_SHOULDER.value)
            left_hip, lhv = _xy(mp_idx.LEFT_HIP.value)
            right_hip, rhv = _xy(mp_idx.RIGHT_HIP.value)

            mapping = {
                0: mp_idx.NOSE.value,
                2: mp_idx.RIGHT_SHOULDER.value,
                3: mp_idx.RIGHT_ELBOW.value,
                4: mp_idx.RIGHT_WRIST.value,
                5: mp_idx.LEFT_SHOULDER.value,
                6: mp_idx.LEFT_ELBOW.value,
                7: mp_idx.LEFT_WRIST.value,
                9: mp_idx.RIGHT_HIP.value,
                10: mp_idx.RIGHT_KNEE.value,
                11: mp_idx.RIGHT_ANKLE.value,
                12: mp_idx.LEFT_HIP.value,
                13: mp_idx.LEFT_KNEE.value,
                14: mp_idx.LEFT_ANKLE.value,
                15: mp_idx.RIGHT_EYE.value,
                16: mp_idx.LEFT_EYE.value,
                17: mp_idx.RIGHT_EAR.value,
            }
            for op_i, mp_i in mapping.items():
                p, v = _xy(mp_i)
                pts[op_i] = [p[0], p[1], max(0.0, min(1.0, v))]

            neck = (left_sh + right_sh) / 2.0
            neck_v = (lsv + rsv) / 2.0
            pts[1] = [neck[0], neck[1], max(0.0, min(1.0, neck_v))]

            midhip = (left_hip + right_hip) / 2.0
            hip_v = (lhv + rhv) / 2.0
            pts[8] = [midhip[0], midhip[1], max(0.0, min(1.0, hip_v))]
    except Exception:
        pass

    # If pose estimation failed or key joints are missing, inject a canonical skeleton
    # to prevent NaNs in HR-VITON agnostic-generation math.
    _core = [1, 2, 5, 8, 9, 12]
    _valid_core = sum(1 for i in _core if pts[i, 2] > 0.1)
    if _valid_core < 4:
        cx = w * 0.5
        y_head = h * 0.14
        y_neck = h * 0.22
        y_sh = h * 0.27
        y_hip = h * 0.56
        y_knee = h * 0.73
        y_ankle = h * 0.90
        sh_dx = w * 0.12
        hip_dx = w * 0.09
        arm_dx = w * 0.18

        pts[:] = 0.0
        pts[0] = [cx, y_head, 1.0]           # nose
        pts[1] = [cx, y_neck, 1.0]           # neck
        pts[2] = [cx - sh_dx, y_sh, 1.0]     # right shoulder (openpose indexing)
        pts[3] = [cx - arm_dx, y_sh + h*0.09, 1.0]
        pts[4] = [cx - arm_dx, y_sh + h*0.18, 1.0]
        pts[5] = [cx + sh_dx, y_sh, 1.0]     # left shoulder
        pts[6] = [cx + arm_dx, y_sh + h*0.09, 1.0]
        pts[7] = [cx + arm_dx, y_sh + h*0.18, 1.0]
        pts[8] = [cx, y_hip, 1.0]            # mid hip
        pts[9] = [cx - hip_dx, y_hip, 1.0]   # right hip
        pts[10] = [cx - hip_dx, y_knee, 1.0]
        pts[11] = [cx - hip_dx, y_ankle, 1.0]
        pts[12] = [cx + hip_dx, y_hip, 1.0]  # left hip
        pts[13] = [cx + hip_dx, y_knee, 1.0]
        pts[14] = [cx + hip_dx, y_ankle, 1.0]
        pts[15] = [cx - w*0.03, y_head - h*0.01, 1.0]
        pts[16] = [cx + w*0.03, y_head - h*0.01, 1.0]
        pts[17] = [cx - w*0.06, y_head + h*0.01, 1.0]

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    edges = [(1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14)]
    for a, b in edges:
        if pts[a, 2] > 0.1 and pts[b, 2] > 0.1:
            p1 = (int(pts[a, 0]), int(pts[a, 1]))
            p2 = (int(pts[b, 0]), int(pts[b, 1]))
            cv2.line(canvas, p1, p2, (255, 255, 255), 4)
    for i in range(18):
        if pts[i, 2] > 0.1:
            cv2.circle(canvas, (int(pts[i, 0]), int(pts[i, 1])), 5, (255, 255, 255), -1)
    cv2.imwrite(str(openpose_img_path), canvas)

    flat = []
    for i in range(18):
        flat.extend([float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])])
    payload = {"version": 1.3, "people": [{"pose_keypoints_2d": flat}]}
    with open(openpose_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _write_parse_maps(person_bgr: np.ndarray, parse_path: Path, parse_agnostic_path: Path, densepose_path: Path):
    """Build coarse fallback parse maps to satisfy HR-VITON input contracts."""
    h, w = person_bgr.shape[:2]
    parse = np.zeros((h, w), dtype=np.uint8)

    person_mask = None
    try:
        rgb = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB)
        with mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1) as seg:
            seg_res = seg.process(rgb)
        if seg_res.segmentation_mask is not None:
            person_mask = (seg_res.segmentation_mask > 0.25).astype(np.uint8)
    except Exception:
        person_mask = None

    if person_mask is None:
        person_mask = np.ones((h, w), dtype=np.uint8)

    ys, xs = np.where(person_mask > 0)
    if len(xs) > 0 and len(ys) > 0:
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        y_mid = int(y0 + 0.55 * (y1 - y0))
        parse[y0:y_mid, x0:x1] = 5   # upper
        parse[y_mid:y1, x0:x1] = 9   # bottom
        parse = parse * person_mask.astype(np.uint8)

    cv2.imwrite(str(parse_path), parse)
    cv2.imwrite(str(parse_agnostic_path), parse)
    cv2.imwrite(str(densepose_path), person_bgr)


def _fit_to_hrviton_canvas(
    img_bgr: np.ndarray,
    target_w: int = 768,
    target_h: int = 1024,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Letterbox image to exact HR-VITON resolution (W=768, H=1024)."""
    h, w = img_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)

    scale = min(target_w / float(w), target_h / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def _looks_corrupt_tryon(output_bgr: np.ndarray) -> bool:
    """Heuristic detector for visibly broken HR-VITON outputs."""
    if output_bgr is None or output_bgr.size == 0:
        return True

    h, w = output_bgr.shape[:2]
    if h < 100 or w < 100:
        return True

    gray = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2GRAY)
    bottom = gray[int(h * 0.72):, :]
    bottom_std = float(bottom.std()) if bottom.size else 0.0

    bright_ratio = float((gray > 245).mean())
    dark_ratio = float((gray < 10).mean())

    # Typical broken generations: large flat bottom blocks + severe washout.
    if bottom_std < 4.0:
        return True
    if bright_ratio > 0.62:
        return True
    if dark_ratio > 0.55:
        return True
    return False


def _render_simple_tryon(person_bgr: np.ndarray, cloth_bgr: np.ndarray, garment_type: str) -> bytes:
    """Deterministic fallback renderer when model output is unusable."""
    base = person_bgr.copy()
    h, w = base.shape[:2]

    gh, gw = cloth_bgr.shape[:2]
    gtype = (garment_type or "").lower()
    # Product images often contain a full-body model; crop to likely garment area.
    if gtype == "bottom":
        cy0, cy1, cx0, cx1 = int(gh * 0.45), int(gh * 0.98), int(gw * 0.18), int(gw * 0.82)
    elif gtype in {"full", "dress"}:
        cy0, cy1, cx0, cx1 = int(gh * 0.16), int(gh * 0.92), int(gw * 0.14), int(gw * 0.86)
    else:
        cy0, cy1, cx0, cx1 = int(gh * 0.18), int(gh * 0.55), int(gw * 0.20), int(gw * 0.80)
    cloth_bgr = cloth_bgr[max(0, cy0):max(1, cy1), max(0, cx0):max(1, cx1)]

    gray = cv2.cvtColor(cloth_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray < 245).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

    # Default target box (x0, y0, x1, y1)
    if gtype == "bottom":
        x0, y0, x1, y1 = int(w * 0.33), int(h * 0.50), int(w * 0.67), int(h * 0.93)
    elif gtype in {"full", "dress"}:
        x0, y0, x1, y1 = int(w * 0.28), int(h * 0.22), int(w * 0.72), int(h * 0.93)
    else:
        x0, y0, x1, y1 = int(w * 0.30), int(h * 0.22), int(w * 0.70), int(h * 0.62)

    # Refine with pose when available.
    try:
        rgb = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)
        with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1) as pose:
            res = pose.process(rgb)
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            sh_l = np.array([lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                             lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y * h])
            sh_r = np.array([lm[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                             lm[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y * h])
            hp_l = np.array([lm[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x * w,
                             lm[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y * h])
            hp_r = np.array([lm[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].x * w,
                             lm[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].y * h])
            center_x = int((sh_l[0] + sh_r[0] + hp_l[0] + hp_r[0]) / 4.0)
            shoulder_w = max(60, int(abs(sh_r[0] - sh_l[0]) * 1.15))
            torso_top = int(min(sh_l[1], sh_r[1]) - h * 0.03)
            torso_mid = int((hp_l[1] + hp_r[1]) / 2.0)

            if gtype == "bottom":
                x0, x1 = center_x - shoulder_w // 2, center_x + shoulder_w // 2
                y0, y1 = torso_mid - int(h * 0.03), int(h * 0.95)
            elif gtype in {"full", "dress"}:
                x0, x1 = center_x - int(shoulder_w * 0.62), center_x + int(shoulder_w * 0.62)
                y0, y1 = torso_top, int(h * 0.95)
            else:
                x0, x1 = center_x - int(shoulder_w * 0.60), center_x + int(shoulder_w * 0.60)
                y0, y1 = torso_top, torso_mid + int(h * 0.04)
    except Exception:
        pass

    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 + 5 or y1 <= y0 + 5:
        x0, y0, x1, y1 = int(w * 0.30), int(h * 0.22), int(w * 0.70), int(h * 0.62)

    tw, th = x1 - x0, y1 - y0
    cloth_rs = cv2.resize(cloth_bgr, (tw, th), interpolation=cv2.INTER_AREA)
    mask_rs = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
    alpha = (mask_rs.astype(np.float32) / 255.0) * 0.88
    alpha = alpha[..., None]

    roi = base[y0:y1, x0:x1].astype(np.float32)
    cloth_f = cloth_rs.astype(np.float32)
    blended = cloth_f * alpha + roi * (1.0 - alpha)
    base[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)

    ok, buf = cv2.imencode(".jpg", base, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError("Fallback try-on renderer failed to encode image.")
    return buf.tobytes()


def _hrviton_python_executable() -> str:
    if HRVITON_PYTHON:
        return HRVITON_PYTHON
    return sys.executable


def _run_dcivton(
    person_img_path,
    cloth_img_path: str,
    progress_cb=None,
) -> tuple[bytes | None, str | None]:
    """Run DCI-VTON inference in-process using the cached model singleton."""
    global _DCIVTON_RUNTIME_MODULE, _DCIVTON_MODEL, _DCIVTON_DEVICE
    if not DCIVTON_ROOT.is_dir():
        return None, f"DCI-VTON repo not found at: {DCIVTON_ROOT}"
    if not os.path.isfile(DCIVTON_CKPT):
        return None, f"DCI-VTON checkpoint not found: {DCIVTON_CKPT}"
    if not os.path.isfile(DCIVTON_INFER):
        return None, f"DCI-VTON infer script not found: {DCIVTON_INFER}"

    try:
        if callable(progress_cb):
            progress_cb(25, "DCI-VTON: loading cached model...")

        env = os.environ.copy()
        env["TF_ENABLE_ONEDNN_OPTS"] = "0"
        env["DCIVTON_CKPT"] = DCIVTON_CKPT
        env["PYTHONPATH"] = str(DCIVTON_ROOT)

        prev_tf = os.environ.get("TF_ENABLE_ONEDNN_OPTS")
        prev_ckpt = os.environ.get("DCIVTON_CKPT")
        prev_pythonpath = os.environ.get("PYTHONPATH")
        prev_cwd = os.getcwd()
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
        os.environ["DCIVTON_CKPT"] = DCIVTON_CKPT
        os.environ["PYTHONPATH"] = str(DCIVTON_ROOT)
        os.chdir(str(DCIVTON_ROOT))

        import importlib.util
        if _DCIVTON_RUNTIME_MODULE is not None:
            dcivton_runtime = _DCIVTON_RUNTIME_MODULE
        else:
            module_name = "_fits_dcivton_runtime"
            spec = importlib.util.spec_from_file_location(module_name, DCIVTON_INFER)
            if spec is None or spec.loader is None:
                return None, f"Unable to load DCI-VTON infer module from: {DCIVTON_INFER}"
            dcivton_runtime = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = dcivton_runtime
            spec.loader.exec_module(dcivton_runtime)
            _DCIVTON_RUNTIME_MODULE = dcivton_runtime

        if _DCIVTON_MODEL is None or _DCIVTON_DEVICE is None:
            _DCIVTON_MODEL, _DCIVTON_DEVICE = dcivton_runtime.get_model()

        if callable(progress_cb):
            progress_cb(55, "DCI-VTON: staging inputs...")
        img_bytes = dcivton_runtime.run_inference_inprocess(
            str(person_img_path),
            cloth_img_path,
            int(DCIVTON_STEPS),
            model=_DCIVTON_MODEL,
            device=_DCIVTON_DEVICE,
        )
        if callable(progress_cb):
            progress_cb(92, "DCI-VTON: decoding output...")
        if not img_bytes:
            return None, "DCI-VTON ran but produced no output image."
        return img_bytes, None

    except subprocess.TimeoutExpired:
        return None, "DCI-VTON timed out (1800s)."
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        if 'prev_tf' in locals():
            if prev_tf is None:
                os.environ.pop("TF_ENABLE_ONEDNN_OPTS", None)
            else:
                os.environ["TF_ENABLE_ONEDNN_OPTS"] = prev_tf
        if 'prev_ckpt' in locals():
            if prev_ckpt is None:
                os.environ.pop("DCIVTON_CKPT", None)
            else:
                os.environ["DCIVTON_CKPT"] = prev_ckpt
        if 'prev_pythonpath' in locals():
            if prev_pythonpath is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = prev_pythonpath
        if 'prev_cwd' in locals():
            try:
                os.chdir(prev_cwd)
            except Exception:
                pass


def _run_hrviton(
    request_dir: Path,
    person_name: str,
    cloth_name: str,
    progress_cb=None,
) -> tuple[bytes | None, str | None]:
    if not HRVITON_ROOT.is_dir():
        return None, f"HR-VITON repo not found at: {HRVITON_ROOT}"

    tocg_ckpt = Path(HRVITON_TOCG_CKPT)
    gen_ckpt = Path(HRVITON_GEN_CKPT)
    if not tocg_ckpt.is_file() or not gen_ckpt.is_file():
        missing = []
        if not tocg_ckpt.is_file():
            missing.append(str(tocg_ckpt))
        if not gen_ckpt.is_file():
            missing.append(str(gen_ckpt))
        return None, "HR-VITON checkpoints missing. Configure HRVITON_TOCG_CKPT and HRVITON_GEN_CKPT. Missing: " + " | ".join(missing)

    output_dir = request_dir / "out"
    _ensure_dir(output_dir)

    if callable(progress_cb):
        progress_cb(65, "Running HR-VITON model inference...")

    use_cuda = (not HRVITON_FORCE_CPU) and torch.cuda.is_available()
    cmd = [
        _hrviton_python_executable(),
        str(HRVITON_ROOT / "test_generator.py"),
        "--occlusion",
        "--cuda", str(bool(use_cuda)),
        "--test_name", "fits_runtime",
        "--tocg_checkpoint", str(tocg_ckpt),
        "--gpu_ids", HRVITON_GPU_IDS,
        "--gen_checkpoint", str(gen_ckpt),
        "--datasetting", "unpaired",
        "--dataroot", str(request_dir),
        "--datamode", "test",
        "--data_list", "test_pairs.txt",
        "--output_dir", str(output_dir),
    ]

    run = subprocess.run(
        cmd,
        cwd=str(HRVITON_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if run.returncode != 0:
        stderr_tail = (run.stderr or run.stdout or "")[-1400:]
        return None, f"HR-VITON inference failed (exit {run.returncode}). {stderr_tail}"

    expected = output_dir / f"{Path(person_name).stem}_{Path(cloth_name).stem}.png"
    if expected.is_file():
        if callable(progress_cb):
            progress_cb(95, "Finalizing try-on output...")
        with open(expected, "rb") as f:
            return f.read(), None

    candidates = sorted(output_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None, "HR-VITON finished without producing an output image."
    if callable(progress_cb):
        progress_cb(95, "Finalizing try-on output...")
    with open(candidates[0], "rb") as f:
        return f.read(), None


def _run_idmvton(
    person_img_path,
    cloth_img_path: str,
    steps: int | None = None,
    width: int | None = None,
    height: int | None = None,
    progress_cb=None,
) -> tuple[bytes | None, str | None]:
    """Run IDM-VTON via an environment-configured command template."""
    if not IDMVTON_ROOT.is_dir():
        return None, f"IDM-VTON repo not found at: {IDMVTON_ROOT}"

    if IDMVTON_USE_WORKER:
        worker_bytes, worker_err = _run_idmvton_worker(
            person_img_path,
            cloth_img_path,
            steps=steps,
            width=width,
            height=height,
            progress_cb=progress_cb,
        )
        if worker_err is None:
            return worker_bytes, None
        if callable(progress_cb):
            progress_cb(75, "IDM worker unavailable, falling back to single-shot mode...")
    if not IDMVTON_COMMAND:
        return None, (
            "IDM-VTON command not configured. Set IDMVTON_COMMAND with placeholders "
            "{python} {root} {person} {cloth} {output} {steps} {width} {height}."
        )

    req_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = IDMVTON_RUNTIME_ROOT / req_id
    _ensure_dir(out_dir)
    out_path = out_dir / "tryon_result.jpg"

    if callable(progress_cb):
        progress_cb(65, "IDM-VTON: running model inference...")

    cmd_text = IDMVTON_COMMAND.format(
        python=IDMVTON_PYTHON,
        root=str(IDMVTON_ROOT),
        person=str(person_img_path),
        cloth=str(cloth_img_path),
        output=str(out_path),
        steps=str(max(1, int(IDMVTON_STEPS if steps is None else steps))),
        width=str(max(256, int(IDMVTON_WIDTH if width is None else width))),
        height=str(max(256, int(IDMVTON_HEIGHT if height is None else height))),
    )

    try:
        cmd = shlex.split(cmd_text, posix=(os.name != "nt"))
    except Exception as exc:
        return None, f"Invalid IDMVTON_COMMAND format: {exc}"

    try:
        run = subprocess.run(
            cmd,
            cwd=str(IDMVTON_ROOT),
            capture_output=True,
            text=True,
            timeout=max(60, int(IDMVTON_TIMEOUT)),
        )
    except subprocess.TimeoutExpired:
        return None, f"IDM-VTON timed out ({IDMVTON_TIMEOUT}s)."
    except Exception as exc:
        return None, f"IDM-VTON launch failed: {type(exc).__name__}: {exc}"

    if run.returncode != 0:
        stderr_tail = (run.stderr or run.stdout or "")[-1800:]
        return None, f"IDM-VTON inference failed (exit {run.returncode}). {stderr_tail}"

    if out_path.is_file():
        if callable(progress_cb):
            progress_cb(95, "Finalizing try-on output...")
        with open(out_path, "rb") as f:
            return f.read(), None

    candidates = sorted(
        list(out_dir.glob("*.png")) + list(out_dir.glob("*.jpg")) + list(out_dir.glob("*.jpeg")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None, "IDM-VTON finished without producing an output image."

    if callable(progress_cb):
        progress_cb(95, "Finalizing try-on output...")
    with open(candidates[0], "rb") as f:
        return f.read(), None


def _run_idmvton_worker(
    person_img_path,
    cloth_img_path: str,
    steps: int | None = None,
    width: int | None = None,
    height: int | None = None,
    progress_cb=None,
) -> tuple[bytes | None, str | None]:
    """Run IDM-VTON through a warm long-lived worker process."""
    if not IDMVTON_WORKER_SCRIPT.is_file():
        return None, f"IDM worker script not found at: {IDMVTON_WORKER_SCRIPT}"

    req_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = IDMVTON_RUNTIME_ROOT / req_id
    _ensure_dir(out_dir)
    out_path = out_dir / "tryon_result.jpg"

    proc = st.session_state.get("_idmvton_worker_proc")
    if proc is None or getattr(proc, "poll", lambda: 1)() is not None:
        try:
            proc = subprocess.Popen(
                [IDMVTON_PYTHON, str(IDMVTON_WORKER_SCRIPT), "--serve"],
                cwd=str(IDMVTON_ROOT),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            st.session_state["_idmvton_worker_proc"] = proc
            st.session_state["_idmvton_worker_warm"] = False
        except Exception as exc:
            return None, f"IDM worker launch failed: {type(exc).__name__}: {exc}"

    if not st.session_state.get("_idmvton_worker_warm", False):
        try:
            _ready = proc.stdout.readline().strip() if proc.stdout else ""
            if _ready:
                st.session_state["_idmvton_worker_warm"] = True
        except Exception:
            pass

    request = {
        "person": str(person_img_path),
        "cloth": str(cloth_img_path),
        "output": str(out_path),
        "steps": int(max(1, int(IDMVTON_STEPS if steps is None else steps))),
        "width": int(max(256, int(IDMVTON_WIDTH if width is None else width))),
        "height": int(max(256, int(IDMVTON_HEIGHT if height is None else height))),
        "seed": 42,
        "garment_desc": "upper body garment",
    }

    if callable(progress_cb):
        progress_cb(66, "IDM worker: running model inference...")

    try:
        if not proc.stdin or not proc.stdout:
            return None, "IDM worker streams are unavailable."
        proc.stdin.write(json.dumps(request) + "\n")
        proc.stdin.flush()
        response_line = proc.stdout.readline().strip()
        if not response_line:
            return None, "IDM worker returned no response."
        response = json.loads(response_line)
    except Exception as exc:
        st.session_state.pop("_idmvton_worker_proc", None)
        st.session_state.pop("_idmvton_worker_warm", None)
        return None, f"IDM worker failed: {type(exc).__name__}: {exc}"

    if not response.get("ok"):
        return None, str(response.get("error") or "IDM worker failed.")

    if out_path.is_file():
        if callable(progress_cb):
            progress_cb(95, "Finalizing try-on output...")
        with open(out_path, "rb") as f:
            return f.read(), None

    return None, "IDM worker completed without producing an output image."


def call_tryon_local(
    person_img_bytes: bytes,
    garment_img_path: str,
    garment_type: str = "full",
    bottom_garment_img_path: str | None = None,
    idmvton_steps: int | None = None,
    idmvton_width: int | None = None,
    idmvton_height: int | None = None,
    progress_cb=None,
):
    """
    Virtual try-on via local HR-VITON runtime.
    Notes:
    - Single-item mode is supported.
    - Combo (top+bottom) is not supported by this adapter.
    Returns (result_bytes, None) on success or (None, error_string) on failure.
    """
    try:
        if callable(progress_cb):
            progress_cb(5, "Preparing inputs...")

        if bottom_garment_img_path:
            return None, "HR-VITON adapter currently supports single-garment try-on only."

        if not os.path.isfile(garment_img_path):
            return None, f"Garment image file not found: {garment_img_path}"

        person_norm, person_err = _normalize_upload_image(person_img_bytes, "Person")
        if person_err:
            return None, person_err

        if callable(progress_cb):
            progress_cb(18, "Normalizing garment image...")

        with open(garment_img_path, "rb") as f:
            garment_bytes = f.read()
        garment_norm, garment_err = _normalize_upload_image(garment_bytes, "Garment")
        if garment_err:
            return None, garment_err

        req_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        req_dir = HRVITON_RUNTIME_ROOT / req_id
        test_dir = req_dir / "test"
        for sub in ["image", "cloth", "cloth-mask", "openpose_img", "openpose_json", "image-parse-v3", "image-parse-agnostic-v3.2", "image-densepose"]:
            _ensure_dir(test_dir / sub)

        person_name = "person.jpg"
        cloth_name = "cloth.jpg"
        person_path = test_dir / "image" / person_name
        cloth_path = test_dir / "cloth" / cloth_name
        cloth_paired_alias_path = test_dir / "cloth" / person_name

        with open(person_path, "wb") as f:
            f.write(person_norm)
        with open(cloth_path, "wb") as f:
            f.write(garment_norm)
        # HR-VITON test loader expects both paired and unpaired cloth names.
        # For single-item runtime we alias paired cloth to the same garment.
        with open(cloth_paired_alias_path, "wb") as f:
            f.write(garment_norm)

        if callable(progress_cb):
            progress_cb(32, "Generating cloth mask...")

        person_bgr = cv2.imdecode(np.frombuffer(person_norm, np.uint8), cv2.IMREAD_COLOR)
        cloth_bgr = cv2.imdecode(np.frombuffer(garment_norm, np.uint8), cv2.IMREAD_COLOR)
        if person_bgr is None or cloth_bgr is None:
            return None, "Failed to decode staged images for HR-VITON preprocessing."

        # Keep original decodes for quality fallback rendering.
        person_bgr_orig = person_bgr.copy()
        cloth_bgr_orig = cloth_bgr.copy()

        # HR-VITON expects portrait-like fixed resolution inputs (W=768, H=1024).
        person_bgr = _fit_to_hrviton_canvas(person_bgr, target_w=768, target_h=1024, bg_color=(0, 0, 0))
        cloth_bgr = _fit_to_hrviton_canvas(cloth_bgr, target_w=768, target_h=1024, bg_color=(255, 255, 255))

        _ok_p, _enc_p = cv2.imencode(".jpg", person_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        _ok_c, _enc_c = cv2.imencode(".jpg", cloth_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not _ok_p or not _ok_c:
            return None, "Failed to encode HR-VITON staged images."

        person_norm = _enc_p.tobytes()
        garment_norm = _enc_c.tobytes()

        with open(person_path, "wb") as f:
            f.write(person_norm)
        with open(cloth_path, "wb") as f:
            f.write(garment_norm)
        # HR-VITON test loader expects both paired and unpaired cloth names.
        # For single-item runtime we alias paired cloth to the same garment.
        with open(cloth_paired_alias_path, "wb") as f:
            f.write(garment_norm)

        _write_cloth_mask_jpg(cloth_bgr, test_dir / "cloth-mask" / cloth_name)
        _write_cloth_mask_jpg(cloth_bgr, test_dir / "cloth-mask" / person_name)
        if callable(progress_cb):
            progress_cb(42, "Estimating pose keypoints...")
        _write_openpose_like(person_bgr, test_dir / "openpose_img" / "person_rendered.png", test_dir / "openpose_json" / "person_keypoints.json")
        if callable(progress_cb):
            progress_cb(55, "Building parse maps...")
        _write_parse_maps(person_bgr, test_dir / "image-parse-v3" / "person.png", test_dir / "image-parse-agnostic-v3.2" / "person.png", test_dir / "image-densepose" / person_name)

        with open(req_dir / "test_pairs.txt", "w", encoding="utf-8") as f:
            f.write(f"{person_name} {cloth_name}\n")

        mode = TRYON_LOCAL_MODE
        if mode == "simple":
            if callable(progress_cb):
                progress_cb(75, "Rendering local try-on...")
            return _render_simple_tryon(person_bgr_orig, cloth_bgr_orig, garment_type), None

        if mode == "dcivton":
            img_bytes, err = _run_dcivton(person_path, str(cloth_path), progress_cb=progress_cb)
            if err:
                if callable(progress_cb):
                    progress_cb(90, "DCI-VTON failed, using fallback compositor...")
                return _render_simple_tryon(person_bgr_orig, cloth_bgr_orig, garment_type), None
            return img_bytes, None

        if mode == "idmvton":
            img_bytes, err = _run_idmvton(
                person_path,
                str(cloth_path),
                steps=idmvton_steps,
                width=idmvton_width,
                height=idmvton_height,
                progress_cb=progress_cb,
            )
            if err:
                return None, err
            return img_bytes, None

        img_bytes, err = _run_hrviton(req_dir, person_name, cloth_name, progress_cb=progress_cb)
        if err:
            if mode == "hrviton":
                return None, err
            if callable(progress_cb):
                progress_cb(90, "HR-VITON failed, applying local fallback compositor...")
            return _render_simple_tryon(person_bgr_orig, cloth_bgr_orig, garment_type), None

        out_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR) if img_bytes else None
        if out_img is None or _looks_corrupt_tryon(out_img):
            if mode == "hrviton":
                if img_bytes:
                    return img_bytes, None
                return None, "HR-VITON produced an unusable image."
            if callable(progress_cb):
                progress_cb(90, "Model output unstable, applying local fallback compositor...")
            return _render_simple_tryon(person_bgr_orig, cloth_bgr_orig, garment_type), None

        return img_bytes, None

    except subprocess.TimeoutExpired:
        return None, "HR-VITON timed out (300s)."
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


TRYON_SAVE_DIR = str((ROOT_DIR / "Tryons").resolve())


def _autosave_tryon(img_bytes: bytes) -> str | None:
    """Save try-on image to TRYON_SAVE_DIR and return the saved path (or None on error)."""
    try:
        os.makedirs(TRYON_SAVE_DIR, exist_ok=True)
        uid  = st.session_state.get("user_id") or "anonymous"
        sid  = st.session_state.get("current_session_id") or "nosession"
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{uid}__{sid}__{ts}.jpg"
        fpath = os.path.join(TRYON_SAVE_DIR, fname)
        with open(fpath, "wb") as fh:
            fh.write(img_bytes)
        return fpath
    except Exception as exc:
        # Non-fatal — never crash the UI over a save failure
        print(f"[FITS] autosave_tryon failed: {exc}")
        return None


def _step_tryon():
    # Keep request latch state across reruns so one click maps to one API call.

    st.markdown(
        "<h1 style='font-size:2rem;font-weight:800;letter-spacing:-0.02em;margin-bottom:0.5rem'>"
        "👗 Virtual Try-On</h1>",
        unsafe_allow_html=True,
    )
    st.caption(f"Local try-on mode: {TRYON_LOCAL_MODE}")

    if TRYON_LOCAL_MODE == "hrviton" and (not Path(HRVITON_TOCG_CKPT).is_file() or not Path(HRVITON_GEN_CKPT).is_file()):
        st.warning(
            "HR-VITON is active, but checkpoints are missing. "
            "Set environment variables HRVITON_TOCG_CKPT and HRVITON_GEN_CKPT to valid .pth files."
        )
    if TRYON_LOCAL_MODE == "idmvton" and not IDMVTON_COMMAND:
        st.warning(
            "IDM-VTON is active, but IDMVTON_COMMAND is not configured. "
            "Set IDMVTON_COMMAND with placeholders {python} {root} {person} {cloth} {output} {steps} {width} {height}."
        )

    item  = st.session_state.get("tryon_item")
    gpath = st.session_state.get("tryon_image")

    def _recover_garment_path(_item: dict, _path: str | None) -> str:
        """Recover a usable local garment image path from item images when state is stale."""
        if _path and os.path.isfile(_path):
            return _path
        _images = (_item or {}).get("images", []) or []
        # Prefer exact basename match first, then first existing image.
        _target_base = os.path.basename(str(_path or ""))
        if _target_base:
            for _p in _images:
                if os.path.basename(str(_p)) == _target_base and os.path.isfile(_p):
                    return _p
        for _p in _images:
            if os.path.isfile(_p):
                return _p
        return str(_path or "")

    def _back_to_recommendations_clear_result():
        st.session_state.tryon_result = None
        st.session_state.step = 5

    def _back_to_recommendations_only():
        st.session_state.step = 5

    def _clear_tryon_override_photo():
        st.session_state.tryon_person_override = None
        st.session_state.tryon_result = None

    def _reselect_garment():
        st.session_state.tryon_result = None
        st.session_state.step = 5

    def _tryon_try_again():
        st.session_state.tryon_result = None

    def _try_different_garment():
        st.session_state.tryon_result = None
        st.session_state.step = 5

    if not item or not gpath:
        st.warning("No garment selected. Go back to Recommendations.")
        st.button("← Back to Recommendations", on_click=_back_to_recommendations_only)
        return

    # Session state can become stale across reruns; recover a valid local image path.
    gpath = _recover_garment_path(item, gpath)
    st.session_state.tryon_image = gpath

    # ── Back button ───────────────────────────────────────────────────────────
    st.button("← Back to Recommendations", use_container_width=False, on_click=_back_to_recommendations_clear_result)

    if st.session_state.tryon_count <= 0:
        st.error("Try-on is currently unavailable.")
        return

    title_text = f"{item.get('brand', '')} — {item.get('name', '')}".strip(" —")
    st.subheader(title_text)

    with st.expander("🔒 Outfit Locks (component swap control)", expanded=False):
        st.caption("Lock any component to prevent accidental replacement when trying on new items.")
        c_top, c_bottom, c_shoes = st.columns(3)
        locks = st.session_state.get("outfit_locks", {"top": False, "bottom": False, "shoes": False})
        with c_top:
            locks["top"] = st.toggle("Top", value=bool(locks.get("top")), key="lock_top")
        with c_bottom:
            locks["bottom"] = st.toggle("Bottom", value=bool(locks.get("bottom")), key="lock_bottom")
        with c_shoes:
            locks["shoes"] = st.toggle("Shoes", value=bool(locks.get("shoes")), key="lock_shoes")
        st.session_state.outfit_locks = locks

        selected = st.session_state.get("outfit_components", {})
        for comp in ("top", "bottom", "shoes"):
            s = selected.get(comp)
            if s:
                st.caption(f"{comp.title()}: {s.get('brand','')} {s.get('name','')} (ID: {s.get('id','')})")

    # ── Person-photo selector ─────────────────────────────────────────────────
    # Priority: explicit override → analysis capture → nothing (prompt separately)
    captures    = st.session_state.get("photo_captures", {})
    # Prefer front shot for try-on; fall back to back shot; then any captured view
    base_bytes  = captures.get("front") or (list(captures.values())[0] if captures else None)
    override    = st.session_state.get("tryon_person_override")
    person_bytes   = override or base_bytes
    using_override = bool(override)
    has_photo      = person_bytes is not None

    # ── If no photo at all, show the capture widget prominently ──────────────
    if not has_photo:
        st.info(
            "📷 **No photo on file.** Take a quick full-body shot below — "
            "it's only used for try-on, your analysis results won't change."
        )
        new_shot = _timer_capture_component("tryon_initial")
        if new_shot is not None:
            st.session_state.tryon_person_override = new_shot.read()
            st.session_state.tryon_result = None
            st.rerun()
        return  # wait until they capture something

    # ── Side-by-side preview ──────────────────────────────────────────────────
    pcol, gcol = st.columns(2)
    with pcol:
        label = "**Your Photo** _(custom)_" if using_override else "**Your Photo** _(from analysis)_"
        st.markdown(label)
        nparr      = np.frombuffer(person_bytes, np.uint8)
        person_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if person_img is None:
            st.error("Could not decode person photo. Please retake or upload a new one.")
            return
        st.image(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB), use_container_width=True)
    with gcol:
        st.markdown("**Garment**")
        try:
            st.image(gpath, use_container_width=True)
        except Exception:
            st.info("Garment image preview unavailable.")

    # ── Optional: retake / swap photo ────────────────────────────────────────
    with st.expander(
        "📷 Use a different photo for try-on"
        + (" ✅ (custom photo active)" if using_override else ""),
        expanded=False,
    ):
        st.caption(
            "Take a new full-body shot here. "
            "Use the 10-second timer to walk into frame first. "
            "Won't affect your analysis results."
        )
        new_shot = _timer_capture_component("tryon_retake")
        if new_shot is not None:
            st.session_state.tryon_person_override = new_shot.read()
            st.session_state.tryon_result = None
            st.rerun()
        if using_override:
            st.button("🗑️ Remove custom photo (use original)", key="clear_tryon_override", on_click=_clear_tryon_override_photo)

    # Use the resolved photo bytes for the API call
    raw_bytes = person_bytes

    ok, err_msg = _validate_tryon_selection(item, gpath)
    if not ok:
        st.error(err_msg)
        st.button("↩ Re-select garment", key="reselect_garment_tryon", on_click=_reselect_garment)
        return

    st.divider()

    # ── Single-item try-on only ─────────────────────────────────────────────
    st.session_state.tryon_mode = "single"
    st.session_state.tryon_combo_bottom_image = None

    # Auto-detect garment type from item category/name for the local try-on backend
    def _infer_garment_type(item: dict) -> str:
        _hay = " ".join([
            str(item.get("category", "") or ""),
            str(item.get("name", "") or ""),
            str(item.get("pants_type", "") or ""),
        ]).lower()
        _LOWER = ("trouser", "trousers", "jeans", "jean", "pant", "pants", "shorts", "shorts", "skirt", "legging")
        _UPPER = ("shirt", "tshirt", "t-shirt", "blouse", "top", "jacket", "coat", "hoodie", "sweater",
                  "sweatshirt", "polo", "vest", "cardigan", "blazer", "pullover", "knitwear")
        _FULL  = ("dress", "jumpsuit", "romper", "playsuit", "overalls", "overall")
        if any(k in _hay for k in _FULL):
            return "full"
        if any(k in _hay for k in _UPPER):
            return "top"
        if any(k in _hay for k in _LOWER):
            return "bottom"
        return "top"  # sensible fallback (single garments rarely need full_body)

    st.session_state["_selected_garment_type"] = _infer_garment_type(item)

    if TRYON_LOCAL_MODE == "idmvton":
        _quality = st.radio(
            "Try-on quality",
            options=["Preview (faster)", "HQ (slower)"],
            horizontal=True,
            key="idmvton_quality_mode",
        )
        if _quality.startswith("Preview"):
            _idm_cfg = {
                "steps": IDMVTON_PREVIEW_STEPS,
                "width": IDMVTON_PREVIEW_WIDTH,
                "height": IDMVTON_PREVIEW_HEIGHT,
                "label": "Preview",
            }
        else:
            _idm_cfg = {
                "steps": IDMVTON_HQ_STEPS,
                "width": IDMVTON_HQ_WIDTH,
                "height": IDMVTON_HQ_HEIGHT,
                "label": "HQ",
            }
        st.caption(
            f"IDM profile: {_idm_cfg['label']} - "
            f"{_idm_cfg['width']}x{_idm_cfg['height']} @ {_idm_cfg['steps']} steps"
        )
    else:
        _idm_cfg = {"steps": None, "width": None, "height": None}

    st.divider()

    # ── Generate button ────────────────────────────────────────────────────── 

    result = st.session_state.get("tryon_result")

    if result is None:
        can_generate = True

        _inflight = bool(st.session_state.get("tryon_request_inflight", False))

        def _queue_tryon_generate():
            _person_sha1 = hashlib.sha1(raw_bytes).hexdigest() if raw_bytes else ""
            _gpath_norm = os.path.normcase(os.path.normpath(str(gpath or "")))
            _gmeta = "missing"
            try:
                if _gpath_norm and os.path.isfile(_gpath_norm):
                    _st = os.stat(_gpath_norm)
                    _gmeta = f"{_st.st_size}:{int(_st.st_mtime_ns)}"
            except Exception:
                _gmeta = "stat_error"
            _sig = "|".join([
                str(item.get("id", "")),
                _gpath_norm,
                _gmeta,
                _person_sha1[:16],
                str(st.session_state.get("_selected_garment_type", "full")),
                str(_idm_cfg.get("steps")),
                str(_idm_cfg.get("width")),
                str(_idm_cfg.get("height")),
            ])
            st.session_state.tryon_generate_request = {
                "sig": _sig,
                "item_id": str(item.get("id", "")),
                "gpath": str(gpath or ""),
                "person_sha1": _person_sha1,
                "quality": str(_idm_cfg.get("label", "")),
            }
            st.session_state.tryon_generate_requested = True

        if can_generate:
            st.button(
                "✨ Generate Try-On",
                type="primary",
                use_container_width=False,
                key="generate_tryon_btn",
                disabled=_inflight,
                on_click=_queue_tryon_generate,
            )

        _req = st.session_state.get("tryon_generate_request") or {}

        if can_generate and st.session_state.get("tryon_generate_requested") and not _inflight:
            st.session_state.tryon_request_inflight = True
            st.session_state.tryon_generate_requested = False

            try:
                _req_sig = str(_req.get("sig", ""))
                _run_gpath = str(_req.get("gpath") or gpath or "")
                _run_person_sha1 = hashlib.sha1(raw_bytes).hexdigest() if raw_bytes else ""
                with st.spinner("Loading Try on…"):
                    _prog_box = st.empty()
                    _prog = _prog_box.progress(0, text="Starting local try-on...")

                    def _on_progress(pct, msg):
                        _p = max(0, min(100, int(pct)))
                        _prog.progress(_p, text=str(msg))

                    img_bytes, err = call_tryon_local(
                        raw_bytes, _run_gpath,
                        garment_type=st.session_state.get("_selected_garment_type", "full"),
                        idmvton_steps=_idm_cfg.get("steps"),
                        idmvton_width=_idm_cfg.get("width"),
                        idmvton_height=_idm_cfg.get("height"),
                        progress_cb=_on_progress,
                    )
                    _prog.progress(100, text="Done")
                    _prog_box.empty()

                display_bytes = img_bytes
                if not err and img_bytes:
                    # Prefer normalized JPEG bytes when OpenCV can decode; otherwise keep
                    # the original payload so browser-native rendering still has a chance.
                    try:
                        _arr = np.frombuffer(img_bytes, np.uint8)
                        _decoded = cv2.imdecode(_arr, cv2.IMREAD_COLOR)
                        if _decoded is not None:
                            # Normalize to JPEG bytes for browser-safe rendering.
                            _ok_jpg, _jpg_buf = cv2.imencode(".jpg", _decoded, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                            if _ok_jpg:
                                display_bytes = _jpg_buf.tobytes()
                    except Exception as _img_exc:
                        # Non-fatal: keep raw bytes and try to render them directly later.
                        st.session_state["_tryon_debug"] = {
                            "decode_warning": str(_img_exc),
                            "result_bytes": len(img_bytes) if img_bytes else 0,
                        }

                _exp_id     = st.session_state.get("current_experiment_id")
                _item_id    = str(st.session_state.get("tryon_expected_item_id", ""))
                _tryon_repo = get_experiment_repository()
                if err:
                    st.session_state["_tryon_debug"] = {
                        "item_id": _item_id,
                        "garment_image": _run_gpath,
                        "person_bytes": len(raw_bytes) if raw_bytes else 0,
                        "person_sha1": _run_person_sha1[:16],
                        "queued_person_sha1": str(_req.get("person_sha1", ""))[:16],
                        "request_sig": _req_sig,
                        "quality": str(_req.get("quality", _idm_cfg.get("label", ""))),
                        "idmvton_steps": _idm_cfg.get("steps"),
                        "idmvton_width": _idm_cfg.get("width"),
                        "idmvton_height": _idm_cfg.get("height"),
                        "result_bytes": len(img_bytes) if img_bytes else 0,
                        "error": err,
                    }
                    if _exp_id and _item_id:
                        _tryon_repo.finalize_tryon(_exp_id, _item_id, "error")
                    st.session_state.tryon_result = {"error": err}
                else:
                    st.session_state.pop("_tryon_debug", None)
                    if _exp_id and _item_id:
                        _tryon_repo.finalize_tryon(_exp_id, _item_id, "success")
                    st.session_state.tryon_count -= 1
                    saved_path = _autosave_tryon(display_bytes)
                    st.session_state.tryon_result = {
                        # Prefer file-path rendering after rerun for reliability.
                        "img": saved_path if saved_path else display_bytes,
                        "img_bytes": display_bytes,
                        "saved_path": saved_path,
                    }
            finally:
                st.session_state.tryon_request_inflight = False
                st.session_state.tryon_generate_request = None

            st.rerun()
    elif "error" in result:
        st.error(f"Try-on failed: {result['error']}")
        dbg = st.session_state.get("_tryon_debug")
        if dbg:
            with st.expander("🔍 Raw API response (debug)"):
                st.json(dbg)
        st.button("🔁 Try again", on_click=_tryon_try_again)
    else:
        st.success("✅ Done!")
        try:
            _img_payload = result.get("img")
            if not _img_payload:
                raise ValueError("Try-on result payload is empty.")

            # If payload is a local file path, render from disk.
            if isinstance(_img_payload, str) and os.path.isfile(_img_payload):
                st.image(_img_payload, caption="Try-On Result", use_container_width=True)
            else:
                if isinstance(_img_payload, str):
                    _img_payload = _img_payload.encode("utf-8", errors="ignore")
                _res_arr = np.frombuffer(_img_payload, np.uint8)
                _res_decoded = cv2.imdecode(_res_arr, cv2.IMREAD_COLOR)
                if _res_decoded is None:
                    # Fallback to Streamlit's native image decoder for non-OpenCV payload variants.
                    st.image(_img_payload, caption="Try-On Result", use_container_width=True)
                else:
                    st.image(cv2.cvtColor(_res_decoded, cv2.COLOR_BGR2RGB), caption="Try-On Result", use_container_width=True)
        except Exception as _display_exc:
            st.error(f"Try-on image could not be displayed: {_display_exc}")
            st.session_state["_tryon_debug"] = {
                "display_error": str(_display_exc),
                "result_bytes": len(result.get("img") or b""),
            }
        dbg = st.session_state.get("_tryon_debug")
        if dbg:
            with st.expander("🔍 Try-on debug"):
                st.json(dbg)
        saved_path = result.get("saved_path")
        if saved_path:
            st.caption(f"💾 Auto-saved → `{saved_path}`")

        _download_payload = result.get("img_bytes")
        if not _download_payload:
            _img_ref = result.get("img")
            if isinstance(_img_ref, str) and os.path.isfile(_img_ref):
                try:
                    with open(_img_ref, "rb") as _fh:
                        _download_payload = _fh.read()
                except Exception:
                    _download_payload = b""
        if not _download_payload:
            _download_payload = b""

        st.download_button(
            "⬇ Download Result",
            data=_download_payload,
            file_name="fits_tryon.jpg",
            mime="image/jpeg",
        )
        st.button("🔁 Try a different garment", on_click=_try_different_garment)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
