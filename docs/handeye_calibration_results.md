# Hand-Eye Calibration Results

Updated: 2026-07-24

This document records the current camera intrinsics, hand-eye calibration result, source data, and relevant code changes for the SI190C arm calibration workflow.

## Camera Intrinsics

Source file:

```text
handeye_calibration/results/camera_intrinsics.npz
```

Image size:

```text
[640, 480]
```

RMS reprojection error:

```text
0.368127
```

Camera matrix K:

```text
[ 593.854106  0.000000  318.217660]
[ 0.000000  590.381992  231.366896]
[ 0.000000  0.000000  1.000000]
```

Distortion coefficients D:

```text
[[-0.09346875]
 [-0.60790846]
 [ 3.10954963]
 [-4.97652265]]
```

## Hand-Eye Result

Recommended method: `park`

Transform file:

```text
handeye_calibration/results/handeye_transform.npy
```

The matrix below is `E_T_C`, camera frame to end-effector frame:

```text
[-0.998905 -0.024562  0.039827 -0.000768]
[ 0.024180 -0.999657 -0.010042  0.046535]
[ 0.040060 -0.009068  0.999156  0.034547]
[ 0.000000  0.000000  0.000000  1.000000]
```

Translation component:

```text
x = -0.77 mm
y = 46.53 mm
z = 34.55 mm
```

## Verification

Calibration data:

```text
handeye_calibration/results/calibration_data.npz
```

Number of pose pairs:

```text
19
```

Verifier result against a stationary ArUco target:

```text
Mean position error: 16.31 mm
Max position error: 22.40 mm
Mean orientation error: 2.19 deg
Max orientation error: 3.87 deg
Per-axis position std: [4.626, 5.003, 4.739] mm
```

Method comparison from the same 19 samples:

```text
tsai:       mean 19.58 mm, mean 5.31 deg
park:       mean 16.31 mm, mean 2.19 deg
horaud:     mean 16.37 mm, mean 2.20 deg
daniilidis: mean 16.99 mm, mean 2.11 deg
```

## Data Distribution

Base-to-end-effector translation norm, min / median / max:

```text
0.0784 / 0.2606 / 0.2914 m
```

Camera-to-ArUco translation norm, min / median / max:

```text
0.2500 / 0.3112 / 0.4214 m
```

Camera-to-ArUco z distance, min / median / max:

```text
0.2390 / 0.2940 / 0.4031 m
```

## Saved Result Files

```text
handeye_calibration/results/camera_intrinsics.npz
handeye_calibration/results/calibration_data.npz
handeye_calibration/results/handeye_transform.npy
handeye_calibration/results/handeye_transform_current_park.npy
handeye_calibration/results/base_to_marker_estimates.npy
handeye_calibration/results/base_to_marker_estimates_current_park.npy
```

## Relevant Code Changes

- `handeye_calibration/handeye_calibration/aruco_detector.py`
  - Tuned ArUco detector parameters and corner refinement.
  - Fixed fisheye intrinsics handling by undistorting marker corners before PnP pose estimation.
  - Added reprojection error output for judging marker pose quality.
  - Fixed fisheye preview axis projection.

- `handeye_calibration/handeye_calibration/data_collector.py`
  - Added multi-frame stable marker capture.
  - A sample is saved only when enough recent frames detect the requested marker.
  - Samples with excessive translation jitter, z-span, or reprojection error are rejected.
