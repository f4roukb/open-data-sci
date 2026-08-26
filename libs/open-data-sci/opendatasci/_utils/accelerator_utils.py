import glob
from functools import lru_cache

# GPU compute nodes (NVIDIA; DRM render nodes, which cover AMD/Intel GPUs;
# ``/dev/dxg``, the GPU paravirtualization device WSL2 exposes instead of
# ``/dev/nvidia*``/``/dev/dri/*``) and NPU nodes (the Linux kernel's ``accel``
# subsystem, e.g. Intel's ``ivpu`` driver). Deliberately excludes
# ``/dev/dri/card*``: those carry display/KMS ioctls, not just compute, and
# are never safe to expose to sandboxed code.
_ACCELERATOR_DEVICE_GLOBS: tuple[str, ...] = (
    "/dev/nvidia*",
    "/dev/dri/renderD*",
    "/dev/dxg",
    "/dev/accel/*",
)


@lru_cache(maxsize=1)
def discover_accelerator_devices() -> list[str]:
    """Existing GPU/NPU compute device nodes on this host (Linux only).

    Cached: hardware doesn't change mid-process, and this runs once per
    sandboxed code execution.
    """
    devices = {path for pattern in _ACCELERATOR_DEVICE_GLOBS for path in glob.glob(pattern)}
    return sorted(devices)
