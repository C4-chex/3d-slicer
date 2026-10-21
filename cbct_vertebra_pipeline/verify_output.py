"""Check spatial integrity and mask consistency; this does NOT verify accuracy."""
import argparse
import json
from pathlib import Path
import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--input', type=Path)
    args = parser.parse_args()
    report = json.loads((args.run / 'run_report.json').read_text(encoding='utf-8'))
    if report['status'] not in ['prediction_requires_review', 'failed_no_vertebrae']:
        raise SystemExit(f"No completed export: {report['status']}")
    source = nib.load(str(args.input or Path(report['input'])))
    merged = nib.load(str(args.run / 'vertebrae_predicted_unverified.nii.gz'))
    data = np.asarray(merged.dataobj)
    assert merged.shape == source.shape
    assert np.allclose(merged.affine, source.affine, atol=1e-4)
    expected_ids = {row['label_id'] for row in report['instances']}
    assert {int(i) for i in np.unique(data) if i} == expected_ids
    ordered_image = nib.load(str(args.run / 'instances_ordered_unverified.nii.gz'))
    ordered = np.asarray(ordered_image.dataobj)
    assert ordered.shape == source.shape and np.allclose(ordered_image.affine, source.affine)
    for row in report['instances']:
        separate = nib.load(str(args.run / 'per_vertebra' / (row['predicted_name'] + '_unverified.nii.gz')))
        assert separate.shape == source.shape
        assert np.allclose(separate.affine, source.affine, atol=1e-4)
        mask = np.asarray(separate.dataobj)
        assert np.array_equal(mask, (data == row['label_id']).astype(np.uint8))
        assert int(mask.sum()) == row['voxel_count']
        assert np.array_equal(ordered == row['instance_id'], mask.astype(bool))
        centroid = nib.affines.apply_affine(source.affine, np.array(np.nonzero(mask)).mean(axis=1))
        assert np.allclose(centroid, [row['centroid_R_mm'], row['centroid_A_mm'], row['centroid_S_mm']], atol=.001)
    result = {'geometry_and_mask_consistency_passed': True, 'checked_instances': len(expected_ids),
              'segmentation_accuracy_verified': False, 'anatomical_numbering_verified': False}
    (args.run / 'geometry_checks.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
