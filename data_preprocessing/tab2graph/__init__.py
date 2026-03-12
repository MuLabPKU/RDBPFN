from . import logger
from . import time_budget
from . import dgl_utils

def _check_dgl_version():
    if not dgl_utils.DGL_AVAILABLE:
        return  # Skip version check if DGL is not installed
    
    import dgl
    required_version = "2.1a240205"
    parts = dgl.__version__.split('+')
    current_version = parts[0]
    if current_version != required_version:
        raise RuntimeError(
            f"Required DGL version {required_version} but the installed version is {current_version}."
        )

#_check_dgl_version()
