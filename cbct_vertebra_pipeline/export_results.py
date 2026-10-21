"""Export reviewable instance masks and physical-coordinate meshes; no accuracy claims."""
from __future__ import annotations
import colorsys
import csv
from pathlib import Path
import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw

ANATOMY = {**{i: f'C{i}' for i in range(1, 8)}, **{i + 7: f'T{i}' for i in range(1, 13)},
           **{i + 19: f'L{i}' for i in range(1, 7)}, 28: 'T13'}


def save_mask(array, reference, path):
    header = reference.header.copy()
    header.set_data_dtype(array.dtype)
    nib.save(nib.Nifti1Image(array, reference.affine, header), str(path))


def overlay(source, labels, output: Path, names: dict[int, str]):
    points = np.nonzero(labels)
    center = [int(np.median(p)) for p in points] if len(points[0]) else [n // 2 for n in source.shape]
    sample = source[::4, ::4, ::4]
    low, high = float(np.percentile(sample, 50) - 350), float(np.percentile(sample, 99) + 200)
    high = max(high, low + 1)
    planes = [('Sagittal', source[center[0], :, :].T[::-1], labels[center[0], :, :].T[::-1]),
              ('Coronal', source[:, center[1], :].T[::-1], labels[:, center[1], :].T[::-1]),
              ('Axial', source[:, :, center[2]].T, labels[:, :, center[2]].T)]
    colors = {i: tuple(int(c * 255) for c in colorsys.hsv_to_rgb((i * .173) % 1, .75, 1)) for i in names}
    width = 1240
    height = 1030 + ((len(names) + 4) // 5) * 26
    canvas = Image.new('RGB', (width, height), '#101820')
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 12), 'SPINEPS on CBCT | MODEL PREDICTIONS ONLY | Contours and anatomical labels require review', fill='white')
    draw.text((20, 34), f'Voxel slice indices: {center}; display window: {low:.0f}..{high:.0f}; stored image axes', fill='#ffcc7c')
    for column, (title, image, mask) in enumerate(planes):
        gray = np.clip((image.astype(np.float32) - low) * 255 / (high - low), 0, 255).astype(np.uint8)
        rgb = np.repeat(gray[..., None], 3, axis=2)
        blended = rgb.copy()
        for label, color in colors.items():
            selected = mask == label
            blended[selected] = (rgb[selected] * .48 + np.asarray(color) * .52).astype(np.uint8)
        xpos = 20 + 410 * column
        draw.text((xpos, 70), title + ' / original', fill='white')
        canvas.paste(Image.fromarray(rgb).resize((400, 400)), (xpos, 95))
        draw.text((xpos, 515), title + ' / prediction', fill='white')
        canvas.paste(Image.fromarray(blended).resize((400, 400)), (xpos, 540))
    for index, (label, name) in enumerate(names.items()):
        x = 20 + index % 5 * 240
        y = 970 + index // 5 * 26
        draw.rectangle((x, y, x + 15, y + 15), fill=colors[label])
        draw.text((x + 22, y), name + ' (unverified)', fill='white')
    canvas.save(output)


def export_results(input_path: Path, output: Path, anatomical_labels: bool, make_mesh=False):
    reference = nib.load(str(input_path))
    raw = nib.load(str(output / 'instances_raw_unverified.nii.gz'))
    semantic = nib.load(str(output / 'semantic_predicted_unverified.nii.gz'))
    for image in (raw, semantic):
        if image.shape != reference.shape or not np.allclose(image.affine, reference.affine, atol=1e-4):
            raise RuntimeError('Output is not on the input voxel grid')
    instances = np.asarray(raw.dataobj).astype(np.uint16)
    subregions = np.asarray(semantic.dataobj)
    # Only retain vertebral bony subregions and valid instance ids, removing
    # discs, sacral/pelvic classes, metal and other structures from delivery.
    allowed = list(ANATOMY) if anatomical_labels else list(range(1, 26)) + [28]
    instances[~np.isin(instances, allowed) | ~np.isin(subregions, list(range(41, 54)))] = 0
    save_mask(instances, reference, output / 'vertebrae_predicted_unverified.nii.gz')
    (output / 'per_vertebra').mkdir(exist_ok=True)
    (output / 'per_instance').mkdir(exist_ok=True)
    rows = []
    names = {}
    voxel_ml = abs(float(np.linalg.det(reference.affine[:3, :3]))) / 1000
    for label in sorted(int(i) for i in np.unique(instances) if i):
        name = ANATOMY[label] if anatomical_labels else f'instance_{label:02d}'
        names[label] = name
        mask = instances == label
        points = np.nonzero(mask)
        lower = np.array([int(p.min()) for p in points])
        upper = np.array([int(p.max()) for p in points])
        centroid_voxel = np.array([float(p.mean()) for p in points])
        centroid_world = nib.affines.apply_affine(reference.affine, centroid_voxel)
        rows.append({'label_id': label, 'predicted_name': name, 'voxel_count': int(mask.sum()),
                     'volume_ml': round(float(mask.sum()) * voxel_ml, 3),
                     'touches_image_border': bool(np.any(lower == 0) or np.any(upper == np.array(mask.shape) - 1)),
                     'centroid_R_mm': round(float(centroid_world[0]), 3),
                     'centroid_A_mm': round(float(centroid_world[1]), 3),
                     'centroid_S_mm': round(float(centroid_world[2]), 3), 'anatomical_name_verified': False})
        save_mask(mask.astype(np.uint8), reference, output / 'per_vertebra' / (name + '_unverified.nii.gz'))
        if make_mesh:
            from skimage.measure import marching_cubes
            import trimesh
            crop = tuple(slice(a, b + 1) for a, b in zip(lower, upper))
            # Padding gives closed surfaces even when an object touches its crop boundary.
            volume = np.pad(mask[crop].astype(np.uint8), 1)
            vertices, faces, _, _ = marching_cubes(volume, .5)
            world = nib.affines.apply_affine(reference.affine, vertices + lower - 1)
            mesh_dir = output / 'meshes_RAS_mm'
            mesh_dir.mkdir(exist_ok=True)
            trimesh.Trimesh(vertices=world, faces=faces, process=False).export(mesh_dir / (name + '_unverified.ply'))
    # Provide neutral spatial instance ids alongside untrusted anatomical names.
    # This is ordering by the supplied NIfTI superior coordinate, not a claim
    # about C/T/L level, and depends on correct input spatial metadata.
    ordered = np.zeros(instances.shape, dtype=np.uint16)
    for number, row in enumerate(sorted(rows, key=lambda item: item['centroid_S_mm'], reverse=True), 1):
        row['instance_id'] = number
        mask = instances == row['label_id']
        ordered[mask] = number
        save_mask(mask.astype(np.uint8), reference, output / 'per_instance' / f'instance_{number:03d}_unverified.nii.gz')
        names[row['label_id']] = f"I{number:02d} / {row['predicted_name']}"
    save_mask(ordered, reference, output / 'instances_ordered_unverified.nii.gz')
    fields = ['instance_id', 'label_id', 'predicted_name', 'voxel_count', 'volume_ml', 'touches_image_border', 'centroid_R_mm', 'centroid_A_mm', 'centroid_S_mm', 'anatomical_name_verified']
    with (output / 'predicted_vertebrae.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    overlay(np.asarray(reference.dataobj), instances, output / 'overlay.png', names)
    saved = nib.load(str(output / 'vertebrae_predicted_unverified.nii.gz'))
    assert saved.shape == reference.shape and np.allclose(saved.affine, reference.affine)
    return {'nonempty_instances': len(rows), 'instances': rows, 'shape_and_affine_match_input': True,
            'pelvic_sacral_disc_and_metal_classes_excluded': True,
            'exclusion_accuracy_verified': False, 'mesh_coordinate_system': 'NIfTI RAS mm' if make_mesh else None,
            'neutral_instance_ordering': 'descending superior world coordinate; anatomical levels unverified'}
