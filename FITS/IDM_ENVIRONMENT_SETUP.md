# IDM-VTON Isolated Environment Setup

## Overview
To avoid dependency conflicts between the Streamlit app and IDM-VTON backend, the try-on backend now runs in a **separate isolated Python environment** (`.venv_idm`).

## Architecture

### Main App Environment (`.venv`)
- **Location**: `FITS/.venv`
- **Purpose**: Run Streamlit app + recommender system
- **Key Dependencies**: streamlit, sentence-transformers, torch, mediapipe, etc.
- **NumPy Version**: 1.26.4 (stable with cv2)

### IDM Backend Environment (`.venv_idm`)
- **Location**: `FITS/IDM-VTON-main/.venv_idm`
- **Purpose**: Run isolated IDM-VTON try-on inference
- **Key Dependencies**: transformers 4.36.2, diffusers, torch, opencv-python, etc.
- **NumPy Version**: 1.26.4 (compatible with transformers 4.36.2)

## How It Works

1. **App startup**: Main environment starts normally with Streamlit
2. **Try-on request**: App spawns subprocess using isolated IDM Python executable
3. **Subprocess execution**: IDM pipeline runs in isolated environment with its own dependencies
4. **Result return**: Output image saved to disk; app reads it back
5. **Isolation maintained**: No cross-environment package conflicts

For best local performance, the app can also keep one isolated IDM worker process alive between requests. In that mode, the model loads once and subsequent try-ons avoid repeating the full import/model startup overhead.

## Configuration

In `fits_app.py`:

```python
# Line ~1324: Auto-detect isolated IDM environment
_IDMVTON_VENV = IDMVTON_ROOT / ".venv_idm" / "Scripts" / "python.exe"
IDMVTON_PYTHON = os.getenv("IDMVTON_PYTHON", 
                            str(_IDMVTON_VENV) if _IDMVTON_VENV.exists() 
                            else sys.executable).strip()
```

### Environment Variables (Optional Overrides)

```bash
# Force different IDM Python (if .venv_idm doesn't exist)
set IDMVTON_PYTHON=C:\path\to\python.exe

# Other IDM config (unchanged)
set IDMVTON_STEPS=30
set IDMVTON_TIMEOUT=1800
set IDMVTON_ROOT=C:\path\to\IDM-VTON-main
```

## Setup Instructions (Already Done)

If setting up fresh:

```bash
# 1. Create isolated environment
cd FITS/IDM-VTON-main
python -m venv .venv_idm

# 2. Create requirements file (requirements_idm.txt is provided)
# Key packages:
# - torch==2.1.2, torchvision==0.16.2
# - transformers==4.36.2, diffusers==0.25.0
# - numpy==1.26.4, opencv-python==4.11.0.86
# - etc. (see requirements_idm.txt)

# 3. Install dependencies
.venv_idm/Scripts/python.exe -m pip install -r requirements_idm.txt
```

## Benefits

✓ **No version conflicts**: Each stack manages its own dependencies  
✓ **Easier debugging**: Problems isolated to specific environment  
✓ **Backend swappable**: Replace IDM with another model in separate env  
✓ **Cleaner app**: Main Streamlit environment stays focused  
✓ **Future-proof**: Different backends can have different Python versions  

## Troubleshooting

**Issue**: "IDM-VTON subprocess failed"  
→ Check that `.venv_idm` exists at: `FITS/IDM-VTON-main/.venv_idm`  
→ Verify `fits_single_infer.py` is in `FITS/IDM-VTON-main/`

**Issue**: "Transformers version mismatch in app"  
→ This is OK! Main app uses its version (4.55.4), IDM uses 4.36.2 in isolated env

**Issue**: Large first-run downloads  
→ Expected: IDM models download on first run (~10GB+); downloads cached locally

**Override to use main environment** (if debugging):
```bash
set IDMVTON_PYTHON=C:\Users\tcq26\Downloads\FITS\FITS\.venv\Scripts\python.exe
```

## File Structure

```
FITS/
├── .venv/                              (Main Streamlit app environment)
│   ├── Scripts/python.exe
│   └── Lib/site-packages/              (App dependencies)
│
├── IDM-VTON-main/
│   ├── .venv_idm/                      (Isolated IDM environment) ← NEW
│   │   ├── Scripts/python.exe          (Used for try-on subprocess)
│   │   └── Lib/site-packages/          (IDM-specific dependencies)
│   ├── fits_single_infer.py            (IDM wrapper script)
│   ├── requirements_idm.txt            (IDM dependencies) ← NEW
│   └── [model files...]
│
└── fits_app.py                         (Updated to use .venv_idm)
```

---
**Last Updated**: 2026-05-10  
**Status**: Active — using isolated IDM environment for all try-on requests
