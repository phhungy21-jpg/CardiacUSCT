"""Phase 3 — ED->ES deformable registration (the load-bearing phase).

Classical SimpleITK B-spline registration, chosen per protocol 3.1 for
transparency/debuggability over a learned method (e.g. VoxelMorph). Produces
one displacement field per patient: ED -> ES. Mean-squares metric is used
because this is intra-patient, same-sequence (monomodal) registration, not
cross-modality.
"""

import numpy as np
import SimpleITK as sitk


def _to_sitk(arr: np.ndarray, spacing: tuple) -> sitk.Image:
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    return img


def register_ed_to_es(ed_arr: np.ndarray, es_arr: np.ndarray, spacing: tuple) -> sitk.Transform:
    """Returns the transform T such that resampling ED with T onto the ES
    grid warps ED into ES space (fixed=ES, moving=ED)."""
    fixed = _to_sitk(es_arr, spacing)
    moving = _to_sitk(ed_arr, spacing)

    mesh_size = [4, 4, 2]
    initial_transform = sitk.BSplineTransformInitializer(fixed, mesh_size)

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.5)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsGradientDescentLineSearch(
        learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10
    )
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    R.SetInitialTransform(initial_transform, inPlace=True)

    # NOTE: a finer multi-resolution BSpline (scaleFactors [1,2,4]) was tried and
    # rejected — 5-7x slower with no Dice improvement (see LOG.md run 2026-07-06-04).
    return R.Execute(fixed, moving)


def warp(arr: np.ndarray, spacing: tuple, ref_arr: np.ndarray, transform: sitk.Transform, is_mask: bool) -> np.ndarray:
    moving = _to_sitk(arr, spacing)
    ref = _to_sitk(ref_arr, spacing)
    interp = sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
    warped = sitk.Resample(moving, ref, transform, interp, 0.0, moving.GetPixelID())
    return sitk.GetArrayFromImage(warped)


def dice_per_label(pred_mask: np.ndarray, true_mask: np.ndarray) -> dict:
    labels = sorted(set(np.unique(true_mask).tolist()) - {0})
    result = {}
    for label in labels:
        pred_bin = pred_mask == label
        true_bin = true_mask == label
        intersection = np.logical_and(pred_bin, true_bin).sum()
        denom = pred_bin.sum() + true_bin.sum()
        result[int(label)] = float(2 * intersection / denom) if denom > 0 else float("nan")
    return result


def surface_distance_per_label(pred_mask: np.ndarray, true_mask: np.ndarray, spacing: tuple) -> dict:
    """Average (symmetric) Hausdorff surface distance in mm, per label. Fairer
    than Dice for thin structures — a 1-voxel boundary error tanks Dice for a
    thin wall far more than it tanks surface distance."""
    labels = sorted(set(np.unique(true_mask).tolist()) - {0})
    result = {}
    for label in labels:
        pred_bin = (pred_mask == label).astype(np.uint8)
        true_bin = (true_mask == label).astype(np.uint8)
        if pred_bin.sum() == 0 or true_bin.sum() == 0:
            result[int(label)] = float("nan")
            continue
        pred_img = _to_sitk(pred_bin, spacing)
        true_img = _to_sitk(true_bin, spacing)
        hd_filter = sitk.HausdorffDistanceImageFilter()
        hd_filter.Execute(true_img, pred_img)
        result[int(label)] = float(hd_filter.GetAverageHausdorffDistance())
    return result


def _signed_distance(mask_arr: np.ndarray, label: int, spacing: tuple) -> sitk.Image:
    bin_img = _to_sitk((mask_arr == label).astype(np.uint8), spacing)
    return sitk.SignedMaurerDistanceMap(
        bin_img, insideIsPositive=False, squaredDistance=False, useImageSpacing=True
    )


def register_ed_to_es_mask_guided(
    ed_arr: np.ndarray, es_arr: np.ndarray, ed_mask: np.ndarray, es_mask: np.ndarray, spacing: tuple,
    label_weights: dict = None,
) -> sitk.Transform:
    """Two-stage: (1) intensity Demons for the global/epicardial alignment,
    (2) warm-started refinement using the sum of signed distance transforms
    across all labels (RV, myocardium, LV) jointly, to fix the large-
    deformation under-recovery seen at the endocardium (see LOG.md run
    2026-07-06-06) without distorting the adjacent myocardium the way an
    LV-only distance target did (see run 2026-07-06-07). Stage 2 directly
    targets the true ES boundaries, so post-hoc Dice against them is no
    longer an independent check for this method — see diffeomorphism_check
    and warped_intensity_residual for the substitute validation."""
    label_weights = label_weights or {1: 1.0, 2: 1.0, 3: 1.0}
    labels = list(label_weights.keys())

    stage1_transform = register_ed_to_es_demons(ed_arr, es_arr, spacing)
    stage1_field = stage1_transform.GetDisplacementField()

    ed_dists = [_signed_distance(ed_mask, label, spacing) * label_weights[label] for label in labels]
    es_dists = [_signed_distance(es_mask, label, spacing) * label_weights[label] for label in labels]
    ed_dist = ed_dists[0]
    for d in ed_dists[1:]:
        ed_dist += d
    es_dist = es_dists[0]
    for d in es_dists[1:]:
        es_dist += d

    demons2 = sitk.DiffeomorphicDemonsRegistrationFilter()
    demons2.SetNumberOfIterations(150)
    demons2.SetStandardDeviations(1.0)
    refined_field = demons2.Execute(es_dist, ed_dist, stage1_field)

    return sitk.DisplacementFieldTransform(refined_field)


def diffeomorphism_check(transform: sitk.Transform, ref_arr: np.ndarray, spacing: tuple) -> dict:
    """Fraction of voxels with non-positive Jacobian determinant (folding —
    physically implausible motion) and the min/max Jacobian determinant."""
    if hasattr(transform, "GetDisplacementField"):
        disp_field = transform.GetDisplacementField()
    else:
        ref = _to_sitk(ref_arr, spacing)
        df_filter = sitk.TransformToDisplacementFieldFilter()
        df_filter.SetReferenceImage(ref)
        disp_field = df_filter.Execute(transform)
    jac_filter = sitk.DisplacementFieldJacobianDeterminantFilter()
    jac = sitk.GetArrayFromImage(jac_filter.Execute(disp_field))
    return {
        "frac_nonpositive_jacobian": float((jac <= 0).mean()),
        "min_jacobian": float(jac.min()),
        "max_jacobian": float(jac.max()),
    }


def warped_intensity_residual(ed_arr: np.ndarray, es_arr: np.ndarray, spacing: tuple, transform: sitk.Transform) -> float:
    """Mean absolute difference between warped ED intensity and true ES
    intensity (both z-scored) — checks overall image alignment didn't
    degrade when the registration was refined to fit mask geometry instead
    of intensity alone."""
    warped_ed = warp(ed_arr, spacing, es_arr, transform, is_mask=False)
    return float(np.abs(warped_ed - es_arr).mean())


def register_ed_to_es_demons(ed_arr: np.ndarray, es_arr: np.ndarray, spacing: tuple) -> sitk.Transform:
    """Diffeomorphic Demons alternative — guarantees smooth, invertible
    deformations, often better for thin structures (RV wall, myocardium)
    than a free-form B-spline."""
    fixed = _to_sitk(es_arr, spacing)
    moving = _to_sitk(ed_arr, spacing)

    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(1024)
    matcher.SetNumberOfMatchPoints(7)
    matcher.ThresholdAtMeanIntensityOn()
    moving_matched = matcher.Execute(moving, fixed)

    demons = sitk.DiffeomorphicDemonsRegistrationFilter()
    demons.SetNumberOfIterations(300)
    demons.SetStandardDeviations(1.5)
    disp_field = demons.Execute(fixed, moving_matched)

    return sitk.DisplacementFieldTransform(disp_field)
