The VoxCPM web demo (`app.py`) and its assets are from OpenBMB/VoxCPM,
commit `f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69`, Apache-2.0.

Upstream: https://github.com/OpenBMB/VoxCPM/tree/f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69

AI-Lab imports `create_demo_interface` and injects its already-loaded
VoxCPM2 model. It does not run the upstream `run_demo` entry point or load
another checkpoint. Two asset paths in `app.py` are changed from the
working directory to the vendored package directory. The seed adapter in
`generate_tts_audio` accommodates the installed VoxCPM 2.0.3 API, whose
`_generate` method has no `seed` parameter. The denoise toggle is disabled
when AI-Lab's installed runtime has no ZipEnhancer checkpoint.
