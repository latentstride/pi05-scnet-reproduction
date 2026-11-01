# Environment reproduction status

Environment setup and inference from a real LIBERO observation have passed. The project uses a shared research account; all assets, Python installations and persistent caches reside in the project owner's personal directory. Public examples use `<PROJECT_ROOT>`. Concrete account paths and infrastructure records remain private.

- OpenPI commit: `215abfb217dbac7d5f1273282331b9b1866c0479`.
- Main environment: Python 3.11.18, JAX/JAXLIB 0.5.3, PyTorch 2.7.1, NumPy 1.26.4; 202 packages pass compatibility checks and core imports pass after relocation.
- Official pi05_libero checkpoint: all 16 files pass CRC32C verification; available MD5 checks pass. The tokenizer loads and encodes from local cache.
- Separate Python 3.8.20 simulation environment: all 125 packages pass compatibility checks. A real LIBERO scene loads with software offscreen rendering, accepts ten settling actions and produces two 224×224 RGB images and an 8-element robot state.
- Official policy loads in 12.4185 seconds and produces a finite 10×7 action sequence. First inference takes 42.2381 seconds including compilation; the second call with the same observation takes 0.100954 seconds. These are two measured calls, not a latency benchmark.

The first inference attempt failed before model loading because the writable sandbox lacked the nvidia-smi command mount point. The runtime now binds that executable explicitly and the successful retry exits with code 0:0.

See [the inference validation report](docs/inference_validation.md) for inputs, outputs and archive locations. Full closed-loop control and the restricted 15-episode evaluation have passed: tasks 0–2, initial states 0–4, all 15 successful. The job completed with exit code 0:0 in 17 minutes 4 seconds. All saved trajectories and video frame counts passed validation. See [the evaluation report](evaluation_report.md). This is a restricted functional baseline, not the full official benchmark.
