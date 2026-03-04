# Copilot Instructions for TensorRT-YOLO

This repository is a TensorRT-based YOLO deployment toolkit for NVIDIA devices, supporting C++ and Python.

## Build, Test, and Lint

### C++ Build
The project uses CMake.

```bash
# Configure
cmake -S . -B build -D TRT_PATH=/path/to/tensorrt -D BUILD_PYTHON=ON -D CMAKE_INSTALL_PREFIX=./install
# Build and Install
cmake --build build -j$(nproc) --config Release --target install
```

### Python Build & Install
```bash
pip install --upgrade build
python -m build --wheel
pip install dist/trtyolo-*-py3-none-any.whl
```

### Testing
There are no dedicated unit tests. Use the examples to verify functionality.

**C++ Example (Detect):**
```bash
# Build example (usually requires building the project first)
cd examples/detect
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/path/to/install/dir
cmake --build .
../bin/detect /path/to/engine /path/to/image
```

**Python Example:**
```bash
python examples/detect/detect.py --engine /path/to/engine --image /path/to/image
```

### Linting
Python code is linted with `ruff`. Configuration is in `pyproject.toml`.
```bash
ruff check .
```

## High-Level Architecture

- **Core Library (`modules/trtyolo`):** Contains the C++ implementation.
  - `infer/`: Inference logic, backend, and the main header `trtyolo.hpp`.
  - `core/`: Memory management and buffer handling.
  - `binding/`: Python bindings using `pybind11`.
- **Plugins (`modules/plugin`):** Custom TensorRT plugins (e.g., EfficientNMS) to accelerate post-processing.
- **Examples (`examples/`):** Task-specific examples (detect, segment, pose, etc.) for both C++ and Python.

## Key Conventions

### C++
- **Single Header:** Users should only include `trtyolo.hpp`.
- **Image Handling:** Use the `trtyolo::Image` struct to wrap image data (pointer, dims, pitch) without copying.
- **Model Classes:** Use specific classes for tasks: `DetectModel`, `SegmentModel`, `ClassifyModel`, `PoseModel`, `OBBModel`.
- **Configuration:** Use `trtyolo::InferOption` to configure inference (e.g., `enableSwapRB()`, `setNormalizeParams()`).

### Python
- **Unified Class:** Use the `TRTYOLO` class from the `trtyolo` package.
- **Task Specification:** Specify the task type (`detect`, `segment`, etc.) in the constructor.
- **Input:** Accepts OpenCV images (numpy arrays).

### General
- **Model Export:** Models should be exported using the `trtyolo-export` tool (separate repo/branch) to be compatible.
- **No Third-Party C++ Runtime Dependencies:** The installed C++ library is designed to be self-contained (except for CUDA/TRT).
