"""Local or GPU-server exploratory CBCT inference using pinned SPINEPS CT weights."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parent


def model_folder(models_dir: Path, directory: str) -> Path:
    matches = list((models_dir / directory).rglob('inference_config.json'))
    if len(matches) != 1:
        raise RuntimeError(f'Run download_models.py first; expected one config under {directory}, got {len(matches)}')
    return matches[0].parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--models-dir', type=Path, default=ROOT / 'models')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--offset', type=float, default=0.0, help='Subtract an exploratory intensity offset; not HU calibration')
    parser.add_argument('--tta', action='store_true', help='Enable mirrored test-time augmentation; slower on CPU')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--no-labeling', action='store_true', help='Debug instance masks without claiming anatomical numbering')
    parser.add_argument('--mesh', action='store_true')
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error('Input NIfTI file does not exist')
    args.output = args.output.resolve()
    if args.output == args.input.parent.resolve():
        parser.error('Choose a separate output directory, not the source directory')
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / 'run_report.json'
    if report_path.exists():
        parser.error('Output already has a run report; use a new output directory')
    os.environ['SPINEPS_SEGMENTOR_MODELS'] = str(args.models_dir.resolve())
    os.environ['MPLCONFIGDIR'] = str(args.output / 'runtime_cache' / 'matplotlib')
    os.environ['TORCH_HOME'] = str(ROOT / 'models' / 'torch_cache')
    os.environ['OMP_NUM_THREADS'] = str(args.threads)
    started = time.monotonic()
    report = {'input': str(args.input.resolve()), 'device': args.device, 'status': 'started',
              'offset_subtracted': args.offset, 'HU_calibrated': False,
              'model_source_modality': 'CT', 'target_modality': 'user-described CBCT',
              'model': 'SPINEPS CT semantic + CT instance + VERIDAH CT labeling',
              'tta': args.tta, 'auto_spine_localizer': 'disabled_for_small_FOV_full_volume_trial',
              'anatomical_labeling_enabled': not args.no_labeling, 'anatomical_labels_verified': False,
              'reference_labels_available': False, 'Dice': None,
              'limitations': ['CT weights on CBCT have not been validated', 'Local field of view may make anatomical numbering ambiguous',
                              'Partial vertebrae and instrument artifacts need visual review', 'This is pretrained inference, not model training']}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    try:
        import nibabel as nib
        import numpy as np
        import torch
        from TPTBox import BIDS_FILE
        from spineps.get_models import get_actual_model
        from spineps.seg_run import segment_image
        from spineps.seg_enums import ErrCode
        from export_results import export_results

        if args.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable. Use a CUDA PyTorch build on the server, or --device cpu for a local test.')
        torch.set_num_threads(args.threads)
        report['versions'] = {name: importlib.metadata.version(name) for name in ['spineps', 'TPTBox', 'torch', 'nnunetv2', 'nibabel']}
        report['gpu_name'] = torch.cuda.get_device_name(0) if args.device == 'cuda' else None
        cpu = args.device == 'cpu'
        semantic = get_actual_model(model_folder(args.models_dir, 'CT_semantic_v2.0.0'), use_cpu=cpu)
        # The upstream CT model normally invokes another CT model to localize/crop
        # the spine. For these ~20 cm CBCTs, process the entire supplied field of
        # view. Only the runtime config is changed; weight files stay untouched.
        semantic.inference_config.needs_corp = False
        semantic.inference_config.inference_augmentation = args.tta
        semantic.load()
        instance = get_actual_model(model_folder(args.models_dir, 'CT_instance_v2.0.0'), use_cpu=cpu).load()
        labeling = None if args.no_labeling else get_actual_model(model_folder(args.models_dir, 'CT_labeling_v1.4.0'), use_cpu=cpu).load()
        report['model_load_seconds'] = round(time.monotonic() - started, 2)
        print('Models loaded', json.dumps(report['versions']), flush=True)
        # Stage into the output tree so upstream derivatives cannot touch input
        # folders. The ct suffix selects the CT-specific input path in SPINEPS.
        staged_dir = args.output / 'work'
        staged_dir.mkdir(exist_ok=True)
        staged = staged_dir / 'sub-trial_ct.nii.gz'
        original = nib.load(str(args.input))
        array = original.get_fdata(dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError('Input contains non-finite voxels')
        array -= args.offset
        staged_header = original.header.copy()
        staged_header.set_data_dtype(np.float32)
        nib.save(nib.Nifti1Image(array, original.affine, staged_header), str(staged))
        del array
        result = segment_image(BIDS_FILE(str(staged), dataset=str(staged_dir), verbose=False),
                               model_semantic=semantic, model_instance=instance, model_labeling=labeling,
                               auto_crop_to_spine=False, proc_sem_n4_bias_correction=False,
                               proc_inst_batch_size=args.batch_size, proc_inst_amp=False,
                               proc_assign_missing_cc_fast=True,
                               return_output_instead_of_save=True, timing=True, verbose=False)
        if len(result) != 4:
            report['pipeline_error_code'] = str(result[-1])
            raise RuntimeError(f'SPINEPS did not return completed masks: {result[-1]}')
        semantic_nii, vertebra_nii, centroids, errcode = result
        if errcode != ErrCode.OK or semantic_nii is None or vertebra_nii is None:
            raise RuntimeError(f'SPINEPS returned {errcode}')
        semantic_nii.save(args.output / 'semantic_predicted_unverified.nii.gz', verbose=False)
        vertebra_nii.save(args.output / 'instances_raw_unverified.nii.gz', verbose=False)
        if centroids is not None:
            centroids.save(args.output / 'centroids_unverified.json', verbose=False)
        export = export_results(args.input, args.output, anatomical_labels=not args.no_labeling, make_mesh=args.mesh)
        report.update(export)
        report['status'] = 'prediction_requires_review' if export['nonempty_instances'] else 'failed_no_vertebrae'
        report['runtime_seconds'] = round(time.monotonic() - started, 2)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        if not export['nonempty_instances']:
            raise SystemExit(2)
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = f'{type(error).__name__}: {error}'
        report['runtime_seconds'] = round(time.monotonic() - started, 2)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        traceback.print_exc()
        raise SystemExit(1) from error


if __name__ == '__main__':
    main()
