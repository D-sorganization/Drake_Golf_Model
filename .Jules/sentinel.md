# Sentinel's Journal

## 2025-12-11 - Secure URDF Loading
**Vulnerability:** URDF generation created a temporary file `golf_model.urdf` in the current working directory. This introduced a potential race condition (if multiple instances run concurrently) and a symlink attack vector in shared environments, as well as polluting the file system.
**Learning:** `pydrake`'s `Parser` class provides `AddModelsFromString` (since at least Drake v1.0), which allows loading model descriptions directly from memory. This is superior to the "write-then-read" pattern.
**Prevention:** When generating model files (URDF/SDF) programmatically, always check if the parser supports string input (`AddModelsFromString` or similar) to avoid insecure temporary file creation.

## 2025-12-12 - Secure Meshcat Binding
**Vulnerability:** The default `StartMeshcat()` or `Meshcat()` configuration in `pydrake` binds to all network interfaces (`host="*"`), potentially exposing the visualization server and its controls (sliders) to the local network or internet if ports are open.
**Learning:** `pydrake`'s `StartMeshcat` helper prioritizes ease of use (and Deepnote compatibility) over security by default. Explicit configuration via `MeshcatParams` is required to restrict access.
**Prevention:** When initializing `Meshcat`, always use `MeshcatParams` with `host="localhost"` (or `127.0.0.1`) for local applications to ensure the server is only accessible from the local machine.
