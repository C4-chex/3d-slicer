"""Create a portable NIfTI inventory. File counts are not patient counts."""
import argparse
import json
from pathlib import Path
import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    paths = sorted(args.input.glob('*.nii.gz')) + sorted(args.input.glob('*.nii'))
    if not paths:
        raise SystemExit('No NIfTI files in the input directory')
    cases = []
    for path in paths:
        image = nib.load(str(path))
        if len(image.shape) != 3:
            raise ValueError(f'{path.name}: expected one 3D volume')
        cases.append({'file': path.name, 'bytes': path.stat().st_size, 'shape': list(image.shape),
                      'spacing_mm': [float(i) for i in image.header.get_zooms()[:3]],
                      'field_of_view_mm': [round(float(s * z), 3) for s, z in zip(image.shape, image.header.get_zooms())],
                      'orientation': list(nib.aff2axcodes(image.affine)),
                      'dtype': str(image.get_data_dtype()), 'reference_label_status': 'not_confirmed'})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'file_count': len(cases), 'independent_patient_count': None,
                                      'reference_annotations_confirmed': False, 'cases': cases}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Inventoried {len(cases)} volumes -> {args.output}')


if __name__ == '__main__':
    main()
