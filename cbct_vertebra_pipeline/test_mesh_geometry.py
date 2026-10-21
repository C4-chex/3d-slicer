"""Protect the physical coordinate frame needed for later CT-CBCT registration."""
import json
from pathlib import Path
import tempfile
import unittest
import nibabel as nib
import numpy as np
import trimesh
from export_results import export_results


class MeshGeometryTest(unittest.TestCase):
    def test_export_uses_original_ras_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            affine = np.array([[-.5, 0, 0, -20], [0, -.75, 0, -40], [0, 0, 1.25, 2], [0, 0, 0, 1.]])
            shape = (14, 16, 18)
            image = nib.Nifti1Image(np.zeros(shape, np.int16), affine)
            nib.save(image, folder / 'input.nii.gz')
            instances = np.zeros(shape, np.uint16)
            semantics = np.zeros(shape, np.uint16)
            instances[3:8, 4:10, 5:12] = 20
            semantics[3:8, 4:10, 5:12] = 49
            nib.save(nib.Nifti1Image(instances, affine), folder / 'instances_raw_unverified.nii.gz')
            nib.save(nib.Nifti1Image(semantics, affine), folder / 'semantic_predicted_unverified.nii.gz')
            result = export_results(folder / 'input.nii.gz', folder, anatomical_labels=True, make_mesh=True)
            self.assertEqual(result['nonempty_instances'], 1)
            mesh = trimesh.load(folder / 'meshes_RAS_mm' / 'L1_unverified.ply', force='mesh')
            # Cube boundary lies half a voxel beyond voxel centers. In RAS mm,
            # the negative x/y axes must reverse which corner gives min/max.
            expected = np.array([[-23.75, -47.125, 7.625], [-21.25, -42.625, 16.375]])
            np.testing.assert_allclose(mesh.bounds, expected, atol=1e-4)
            self.assertTrue(mesh.is_watertight)
            delivered = nib.load(folder / 'vertebrae_predicted_unverified.nii.gz')
            np.testing.assert_allclose(delivered.affine, affine)
            np.testing.assert_array_equal(np.asarray(delivered.dataobj), instances)


if __name__ == '__main__':
    unittest.main()
