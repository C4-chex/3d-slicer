#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Start from a server image with Python 3.11/3.12 and a WORKING CUDA PyTorch
# >=2.8. We deliberately preserve its torch build and CUDA runtime.
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -c 'import sys,torch; assert (3,11)<=sys.version_info[:2]<(3,13), "Use Python 3.11 or 3.12"; assert tuple(map(int,torch.__version__.split("+")[0].split(".")[:2])) >= (2,8), "Use a server image with PyTorch >=2.8"; print("Existing torch:",torch.__version__,"CUDA:",torch.cuda.is_available()); assert torch.cuda.is_available(), "The selected server image must have working CUDA PyTorch"'
"$PYTHON_BIN" -m venv --system-site-packages .venv
source .venv/bin/activate
python - <<'PY'
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
pins = []
for package in ('torch', 'torchvision', 'torchaudio'):
    try:
        pins.append(f'{package}=={version(package)}')
    except PackageNotFoundError:
        pass
Path('runtime_torch_constraints.txt').write_text('\n'.join(pins) + '\n', encoding='utf-8')
PY
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}" python -m pip install -r requirements.txt -c constraints.txt -c runtime_torch_constraints.txt
python -m pip check
python preflight.py --device cuda
printf '\nEnvironment ready. Next: python download_models.py\n'
