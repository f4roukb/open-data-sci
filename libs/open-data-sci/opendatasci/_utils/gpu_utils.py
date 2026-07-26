import glob
import os

def discover_gpu_devices() -> list[str]:
    """Existing GPU compute device nodes on this host (Linux only).

    Deliberately excludes ``/dev/dri/card*``: those carry display/KMS ioctls,
    not just compute, and are never safe to expose to sandboxed code.
    """
    patterns = ["/dev/nvidia*", "/dev/dri/renderD*"]
    devices: list[str] = []
    for pattern in patterns:
        devices.extend(path for path in glob.glob(pattern) if os.path.exists(path))
    return sorted(set(devices))
