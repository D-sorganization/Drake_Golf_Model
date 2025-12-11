# Sentinel's Journal

## 2025-12-11 - Secure URDF Loading
**Vulnerability:** URDF generation created a temporary file `golf_model.urdf` in the current working directory. This introduced a potential race condition (if multiple instances run concurrently) and a symlink attack vector in shared environments, as well as polluting the file system.
**Learning:** `pydrake`'s `Parser` class provides `AddModelsFromString` (since at least Drake v1.0), which allows loading model descriptions directly from memory. This is superior to the "write-then-read" pattern.
**Prevention:** When generating model files (URDF/SDF) programmatically, always check if the parser supports string input (`AddModelsFromString` or similar) to avoid insecure temporary file creation.
