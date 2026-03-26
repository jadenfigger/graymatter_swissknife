import logging
import numpy as np
import nibabel as nib


def save_estimations_as_nifti(estimations, model, powder_average_path, mask_path, out_path, optimization_method):
    aff, hdr = nib.load(powder_average_path).affine, nib.load(powder_average_path).header

    powder_average = nib.load(powder_average_path).get_fdata()
    if powder_average.ndim == 4:
        powder_average = np.sum(powder_average, axis=-1)
    
    if mask_path is not None:
        # threshold for the mask (coulb also be 0.33)
        # If you touch it, please change it also in powderaverage.py
        mask_threshold = 0 
        mask = nib.load(mask_path).get_fdata()
        mask = (mask > mask_threshold).astype(bool)
        mask = mask & (~np.isnan(powder_average))
    else:
        mask = ~np.isnan(powder_average)
    param_map_shape = mask.shape

    param_names = model.param_names
    # Remove the last parameter (sigma) from the parameter names if the model has a Rician mean correction
    if model.has_noise_correction:
        param_names = param_names[:-1]
    for i, param_name in enumerate(param_names):
        param_map = np.zeros(param_map_shape) * np.nan
        param_map[mask] = estimations[:, i]
        param_map_nifti = nib.Nifti1Image(param_map, aff, hdr)
        if optimization_method == 'nls':
            nib.save(param_map_nifti, f'{out_path}/{model.name.lower()}_{param_name.lower()}.nii.gz')
            logging.info(f'{model.name.lower()}_{param_name.lower()}.nii.gz saved in {out_path}')
        else:
            nib.save(param_map_nifti, f'{out_path}/{optimization_method}_{model.name.lower()}_{param_name.lower()}.nii.gz')
            logging.info(f'{optimization_method}_{model.name.lower()}_{param_name.lower()}.nii.gz saved in {out_path}')


def save_initialization_as_nifti(initializations, model, powder_average_path, mask_path, out_path):
    aff, hdr = nib.load(powder_average_path).affine, nib.load(powder_average_path).header

    powder_average = nib.load(powder_average_path).get_fdata()
    if powder_average.ndim == 4:
        powder_average = np.sum(powder_average, axis=-1)
    
    if mask_path is not None:
        # threshold for the mask (coulb also be 0.33)
        # If you touch it, please change it also in powderaverage.py
        mask_threshold = 0 
        mask = nib.load(mask_path).get_fdata()
        mask = (mask > mask_threshold).astype(bool)
        mask = mask & (~np.isnan(powder_average))
    else:
        mask = ~np.isnan(powder_average)
    param_map_shape = mask.shape

    param_names = model.param_names
    # Remove the last parameter (sigma) from the parameter names if the model has a Rician mean correction
    if model.has_noise_correction:
        param_names = param_names[:-1]
    for i, param_name in enumerate(param_names):
        param_map = np.zeros(param_map_shape) * np.nan
        param_map[mask] = initializations[:, i]
        param_map_nifti = nib.Nifti1Image(param_map, aff, hdr)
        nib.save(param_map_nifti, f'{out_path}/{model.name.lower()}_{param_name.lower()}_initialization.nii.gz')
        logging.info(f'{model.name.lower()}_{param_name.lower()}_initialization.nii.gz saved in {out_path}')


def save_diagnostics_as_nifti(estimations, signal, sigma, microstruct_model, acq_param, 
                               powder_average_path, mask_path, out_path, n_cores=-1):
    """
    Compute and save comprehensive diagnostic maps after model fitting.
    
    Saves:
      - predicted signal (4D NIfTI matching powder average shape)
      - residual map (measured - predicted, 4D)
      - RMSE map (3D, per-voxel root mean squared error)
      - NRMSE map (3D, RMSE normalized by mean measured signal)
      - cost map (3D, final sum-of-squares at each voxel)
      - bounds_hit map (3D, integer encoding which params hit bounds)
      - sigma map (3D, the per-voxel noise used in Rice correction, if applicable)
    """
    from joblib import Parallel, delayed
    from tqdm import tqdm

    logging.info("Computing diagnostics...")

    aff, hdr = nib.load(powder_average_path).affine, nib.load(powder_average_path).header
    powder_average = nib.load(powder_average_path).get_fdata()
    if powder_average.ndim == 4:
        pa_sum = np.sum(powder_average, axis=-1)
    else:
        pa_sum = powder_average
    
    if mask_path is not None:
        mask_threshold = 0
        mask = nib.load(mask_path).get_fdata()
        mask = (mask > mask_threshold).astype(bool)
        mask = mask & (~np.isnan(pa_sum))
    else:
        mask = ~np.isnan(pa_sum)

    param_map_shape = mask.shape
    voxel_nb = len(signal)
    n_measurements = signal.shape[1]

    # --- 1. Compute predicted signal for every voxel ---
    logging.info("  Computing predicted signals...")
    predicted_signal = np.array(
        Parallel(n_jobs=n_cores)(
            delayed(microstruct_model.get_signal)(estimations[i], acq_param)
            for i in tqdm(range(voxel_nb), desc="  Predicted signal")
        )
    )

    # --- 2. Residuals ---
    residuals = signal - predicted_signal

    # --- 3. Per-voxel metrics ---
    mse_per_voxel = np.mean(residuals ** 2, axis=1)
    rmse_per_voxel = np.sqrt(mse_per_voxel)
    mean_signal = np.mean(signal, axis=1)
    mean_signal[mean_signal == 0] = np.nan
    nrmse_per_voxel = rmse_per_voxel / mean_signal
    cost_per_voxel = np.sum(residuals ** 2, axis=1)

    # --- 4. Bounds-hit map ---
    # Integer encoding: bit 0 = param 0 hit bound, bit 1 = param 1, etc.
    n_params = microstruct_model.n_params
    if microstruct_model.has_noise_correction:
        n_check = n_params - 1  # skip sigma
    else:
        n_check = n_params
    param_lim = microstruct_model.param_lim

    bounds_hit = np.zeros(voxel_nb, dtype=np.int32)
    for p in range(n_check):
        lo, hi = param_lim[p][0], param_lim[p][1]
        if lo == hi:
            continue  # fixed parameter, skip
        at_lo = np.abs(estimations[:, p] - lo) < 1e-10 * (hi - lo + 1e-30)
        at_hi = np.abs(estimations[:, p] - hi) < 1e-10 * (hi - lo + 1e-30)
        bounds_hit[at_lo | at_hi] |= (1 << p)

    # Also save per-parameter bound hit maps for easy visualization
    param_names = microstruct_model.param_names
    if microstruct_model.has_noise_correction:
        param_names = param_names[:-1]

    # --- 5. Save everything as NIfTI ---
    prefix = f'{out_path}/diag_{microstruct_model.name.lower()}'

    # RMSE map
    rmse_map = np.zeros(param_map_shape) * np.nan
    rmse_map[mask] = rmse_per_voxel
    nib.save(nib.Nifti1Image(rmse_map, aff, hdr), f'{prefix}_rmse.nii.gz')
    logging.info(f"  Saved {prefix}_rmse.nii.gz")

    # NRMSE map
    nrmse_map = np.zeros(param_map_shape) * np.nan
    nrmse_map[mask] = nrmse_per_voxel
    nib.save(nib.Nifti1Image(nrmse_map, aff, hdr), f'{prefix}_nrmse.nii.gz')
    logging.info(f"  Saved {prefix}_nrmse.nii.gz")

    # Cost (sum of squares) map
    cost_map = np.zeros(param_map_shape) * np.nan
    cost_map[mask] = cost_per_voxel
    nib.save(nib.Nifti1Image(cost_map, aff, hdr), f'{prefix}_cost.nii.gz')
    logging.info(f"  Saved {prefix}_cost.nii.gz")

    # Bounds hit (integer-coded) map
    bounds_map = np.zeros(param_map_shape, dtype=np.int32)
    bounds_map[mask] = bounds_hit
    nib.save(nib.Nifti1Image(bounds_map, aff, hdr), f'{prefix}_bounds_hit.nii.gz')
    logging.info(f"  Saved {prefix}_bounds_hit.nii.gz  (bit0=tex, bit1=Di, bit2=De, bit3=f)")

    # Per-parameter bound hit maps (easier to visualize than the bit-coded version)
    for p, pname in enumerate(param_names):
        lo, hi = param_lim[p][0], param_lim[p][1]
        if lo == hi:
            continue
        hit_map = np.zeros(param_map_shape) * np.nan
        at_lo = np.abs(estimations[:, p] - lo) < 1e-10 * (hi - lo + 1e-30)
        at_hi = np.abs(estimations[:, p] - hi) < 1e-10 * (hi - lo + 1e-30)
        # -1 = hit lower, 0 = interior, +1 = hit upper
        vals = np.zeros(voxel_nb, dtype=np.float32)
        vals[at_lo] = -1.0
        vals[at_hi] = 1.0
        hit_map[mask] = vals
        nib.save(nib.Nifti1Image(hit_map, aff, hdr), f'{prefix}_bounds_{pname.lower()}.nii.gz')
    logging.info(f"  Saved per-parameter bound hit maps")

    # Predicted signal (4D)
    pred_4d = np.zeros(param_map_shape + (n_measurements,)) * np.nan
    pred_4d[mask] = predicted_signal
    nib.save(nib.Nifti1Image(pred_4d, aff, hdr), f'{prefix}_predicted_signal.nii.gz')
    logging.info(f"  Saved {prefix}_predicted_signal.nii.gz")

    # Residual map (4D)
    resid_4d = np.zeros(param_map_shape + (n_measurements,)) * np.nan
    resid_4d[mask] = residuals
    nib.save(nib.Nifti1Image(resid_4d, aff, hdr), f'{prefix}_residuals.nii.gz')
    logging.info(f"  Saved {prefix}_residuals.nii.gz")

    # Measured signal (4D) - save the powder averaged data that was actually fit
    meas_4d = np.zeros(param_map_shape + (n_measurements,)) * np.nan
    meas_4d[mask] = signal
    nib.save(nib.Nifti1Image(meas_4d, aff, hdr), f'{prefix}_measured_signal.nii.gz')
    logging.info(f"  Saved {prefix}_measured_signal.nii.gz")

    # Sigma map (if Rician correction was used)
    if sigma is not None:
        sigma_map = np.zeros(param_map_shape) * np.nan
        sigma_map[mask] = sigma
        nib.save(nib.Nifti1Image(sigma_map, aff, hdr), f'{prefix}_sigma.nii.gz')
        logging.info(f"  Saved {prefix}_sigma.nii.gz")

    # --- 6. Save raw arrays as npz for detailed single-voxel inspection ---
    np.savez_compressed(
        f'{prefix}_raw_arrays.npz',
        estimations=estimations,
        signal=signal,
        predicted_signal=predicted_signal,
        residuals=residuals,
        rmse=rmse_per_voxel,
        nrmse=nrmse_per_voxel,
        cost=cost_per_voxel,
        bounds_hit=bounds_hit,
        bvals=acq_param.b,
        delta=acq_param.delta,
        small_delta=acq_param.small_delta,
        param_names=param_names,
        param_lim=np.array(param_lim[:n_check]),
    )
    logging.info(f"  Saved {prefix}_raw_arrays.npz (for single-voxel analysis)")

    logging.info("Diagnostics complete.")
