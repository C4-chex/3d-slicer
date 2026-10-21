"""Build a code-only upload package. Medical images, weights and environments are excluded."""
import argparse
import hashlib
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT.parent / 'output' / 'cbct_server_starter.zip')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    allowed = {'.py', '.sh', '.md'}
    files = sorted(p for p in ROOT.iterdir() if p.is_file() and (p.suffix in allowed or p.name in ['requirements.txt', 'constraints.txt', 'model_manifest.json', '.gitignore']))
    with zipfile.ZipFile(args.output, 'w', compression=zipfile.ZIP_DEFLATED) as package:
        for path in files:
            info = zipfile.ZipInfo.from_file(path, arcname='cbct_vertebra_pipeline/' + path.name)
            if path.suffix == '.sh':
                info.create_system = 3
                info.external_attr = (0o100755 << 16)
            package.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(args.output) as package:
        assert package.testzip() is None
        assert all('/models/' not in name and '/runs/' not in name and not name.endswith(('.nii', '.nii.gz', '.pth', '.ckpt')) for name in package.namelist())
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix('.zip.sha256').write_text(digest + '  ' + args.output.name + '\n', encoding='ascii')
    print(f'Code-only package: {args.output}\nFiles: {len(files)}\nBytes: {args.output.stat().st_size}\nSHA256: {digest}')


if __name__ == '__main__':
    main()
