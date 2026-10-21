"""Download pinned official weights; never sends medical images to a server."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import zipfile
import requests

ROOT = Path(__file__).resolve().parent


def download_one(model: dict, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    folder = destination / model['directory']
    marker = folder / 'download_verified.json'
    if marker.exists() and next(folder.rglob('inference_config.json'), None):
        print(f"{model['key']}: verified model already present", flush=True)
        return
    archive = destination / model['asset_name']
    partial = destination / (model['asset_name'] + '.part')
    api_url = f"https://api.github.com/repos/Hendrik-code/spineps/releases/assets/{model['asset_id']}"
    expected_hash = model['sha256']
    # Read the official asset digest for older assets when it is available.
    if expected_hash is None:
        metadata = requests.get(api_url, timeout=(15, 30))
        metadata.raise_for_status()
        digest = metadata.json().get('digest')
        expected_hash = digest.split(':', 1)[1] if digest and digest.startswith('sha256:') else None
    start = time.monotonic()
    checksum = hashlib.sha256()
    if archive.exists():
        with archive.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                checksum.update(chunk)
        valid = archive.stat().st_size == model['size'] and (not expected_hash or checksum.hexdigest() == expected_hash)
    else:
        valid = False
    if not valid:
        checksum = hashlib.sha256()
        count = 0
        last_progress = 0
        print(f"{model['key']}: downloading {model['size'] / 1024**2:.1f} MiB", flush=True)
        with requests.get(api_url, params={'download': '1'}, headers={'Accept': 'application/octet-stream'},
                          stream=True, timeout=(20, 60)) as response:
            response.raise_for_status()
            if 'json' in response.headers.get('content-type', ''):
                raise RuntimeError('GitHub returned metadata instead of an archive')
            with partial.open('wb') as stream:
                for chunk in response.iter_content(1024 * 1024):
                    stream.write(chunk)
                    checksum.update(chunk)
                    count += len(chunk)
                    if count - last_progress >= 100 * 1024 * 1024:
                        print(f"{model['key']}: {count / 1024**2:.0f} MiB", flush=True)
                        last_progress = count
        if count != model['size']:
            raise RuntimeError(f"{model['key']}: size mismatch")
        if expected_hash and checksum.hexdigest() != expected_hash:
            raise RuntimeError(f"{model['key']}: SHA256 mismatch")
        partial.replace(archive)
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for item in bundle.infolist():
            if not (folder / item.filename).resolve().is_relative_to(folder.resolve()):
                raise RuntimeError('Unexpected archive member path')
        if bundle.testzip() is not None:
            raise RuntimeError('Archive CRC check failed')
        bundle.extractall(folder)
    # Upstream archives sometimes wrap the model in one directory. Keep that
    # wrapper: SPINEPS resolves inference_config.json recursively.
    if next(folder.rglob('inference_config.json'), None) is None:
        raise RuntimeError(f"{model['key']}: inference config missing")
    marker.write_text(json.dumps({'source': api_url, 'release': model['tag'], 'bytes': model['size'],
                                 'sha256': checksum.hexdigest(), 'official_sha256_checked': expected_hash is not None,
                                 'download_extract_seconds': round(time.monotonic() - start, 1)}, indent=2), encoding='utf-8')
    print(f"{model['key']}: downloaded and verified", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models-dir', type=Path, default=ROOT / 'models')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--only', nargs='+', choices=['ct', 'ct_instance', 'ct_labeling'])
    args = parser.parse_args()
    models = json.loads((ROOT / 'model_manifest.json').read_text(encoding='utf-8'))['models']
    if args.only:
        models = [m for m in models if m['key'] in args.only]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(lambda model: download_one(model, args.models_dir.resolve()), models))


if __name__ == '__main__':
    main()
