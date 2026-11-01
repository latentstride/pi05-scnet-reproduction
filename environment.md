# Environment reproduction status

The project uses a shared research account. All project assets have been moved into the owner's personal project directory. Public examples use `<PROJECT_ROOT>`; concrete account paths, host details and billing records are kept in private deployment notes.

Validated components:

- OpenPI commit: `215abfb217dbac7d5f1273282331b9b1866c0479`.
- Python 3.11.18; JAX 0.5.3; PyTorch 2.7.1+cu126.
- Main environment: 202 packages pass dependency compatibility checks; core configuration import passed before relocation and is being rechecked after relocation.
- Official pi05_libero checkpoint: all 16 files pass CRC32C checks; available MD5 checks also pass.
- Tokenizer loads from local cache and encodes successfully.
- Separate Python 3.8.20 simulation environment: all 125 packages pass compatibility checks after dependency repair.
- A real LIBERO spatial task observation has been captured using software offscreen rendering after relocation: state shape `(8,)`, two uint8 images shaped `(224, 224, 3)`, ten settling actions, fixed initial state and seed.

Migration repairs include interpreter links, editable installation paths and simulator configuration. Python installations and persistent package/framework caches are explicitly scoped to the personal project directory. See the scripts for relative project layout; adapt platform-specific scheduler settings privately.

Model inference, full closed-loop control and the 15-episode evaluation are still pending. GPU hardware probing alone does not establish pi0.5 inference success. Synthetic examples must be recorded separately from real simulator observations.
