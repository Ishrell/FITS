# IDM-VTON setup for FITS

This app now supports `TRYON_LOCAL_MODE=idmvton`.

The app is configured to use `IDM-VTON-main` by default and invokes:

`{python} fits_single_infer.py --person {person} --cloth {cloth} --output {output} --steps {steps} --width {width} --height {height}`

from inside the IDM-VTON repo.

## Required environment variables

- `TRYON_LOCAL_MODE=idmvton`
- `IDMVTON_ROOT` = absolute path to your IDM-VTON repository
- `IDMVTON_COMMAND` = command template used to run inference (optional if using the default wrapper)

Optional:

- `IDMVTON_PYTHON` (default: current Python)
- `IDMVTON_STEPS` (default: `30`)
- `IDMVTON_WIDTH` / `IDMVTON_HEIGHT` (default: `768` / `1024`)
- `IDMVTON_USE_WORKER` (default: `1`; keeps a warm local worker alive between try-on runs)
- `IDMVTON_TIMEOUT` (default: `1800` seconds)
- `IDMVTON_RUNTIME_ROOT` (default: `FITS/Tryons/idmvton_runtime`)

## Command template placeholders

Your `IDMVTON_COMMAND` can use these placeholders:

- `{python}`
- `{root}`
- `{person}`
- `{cloth}`
- `{output}`
- `{steps}`
- `{width}`
- `{height}`

The app replaces placeholders at runtime and executes the resulting command in `IDMVTON_ROOT`.

## Local performance mode

For the fastest local runs, the app prefers a long-lived worker script at `IDM-VTON-main/fits_idm_worker.py` when `IDMVTON_USE_WORKER=1`.

That worker keeps the model hot in memory, so the first request pays the load cost and later runs are much faster.

## Example (template pattern)

Use your repo's actual script and argument names. Example shape:

`{python} inference.py --person {person} --cloth {cloth} --output {output} --steps {steps}`

If your repo expects different argument names, adjust the template accordingly.

## PowerShell launch example

```powershell
Set-Location C:/Users/tcq26/Downloads/FITS/FITS
$env:TRYON_LOCAL_MODE='idmvton'
$env:IDMVTON_ROOT='C:/path/to/IDM-VTON-main'
$env:IDMVTON_STEPS='30'
streamlit run fits_app.py
```

## Notes

- `IDMVTON_COMMAND` must produce an output image at `{output}`, or in the run output folder.
- On CPU, diffusion try-on remains slow; GPU is strongly recommended.
