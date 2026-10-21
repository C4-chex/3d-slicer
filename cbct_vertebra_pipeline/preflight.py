"""Print a hardware/software report without reading or uploading medical images."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--require-models', action='store_true')
    args = parser.parse_args()
    import torch
    from spineps import get_semantic_model, get_instance_model, get_labeling_model
    report = {'python': sys.version.split()[0], 'platform': platform.system(), 'logical_cpus': os.cpu_count(),
              'versions': {p: importlib.metadata.version(p) for p in ['torch', 'spineps', 'TPTBox', 'nnunetv2', 'monai']},
              'cuda_available': torch.cuda.is_available(), 'cuda_runtime': torch.version.cuda,
              'disk_free_GiB': round(shutil.disk_usage(ROOT).free / 1024**3, 1)}
    if torch.cuda.is_available():
        report['gpu'] = torch.cuda.get_device_name(0)
        report['vram_GiB'] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
        check = torch.ones((8, 8, 8), device='cuda').sum()
        assert check.item() == 512
    if args.require_models:
        manifest = json.loads((ROOT / 'model_manifest.json').read_text())
        missing = [m['key'] for m in manifest['models'] if not (ROOT / 'models' / m['directory'] / 'download_verified.json').is_file()]
        report['missing_models'] = missing
        if missing:
            print(json.dumps(report, indent=2))
            raise SystemExit('Run download_models.py before inference')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA is unavailable; choose a CUDA PyTorch image or explicitly use --device cpu')


if __name__ == '__main__':
    main()
