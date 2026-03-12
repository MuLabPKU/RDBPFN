"""DGL availability checker and optional imports."""

try:
    import dgl
    import dgl.graphbolt
    DGL_AVAILABLE = True
    DGL_VERSION = dgl.__version__
except ImportError:
    DGL_AVAILABLE = False
    DGL_VERSION = None
    dgl = None

def check_dgl_required():
    """Raise an error if DGL is required but not available."""
    if not DGL_AVAILABLE:
        raise ImportError(
            "DGL is required for graph-based functionality but is not installed. "
            "Please install DGL to use graph-based solutions and datasets."
        )

def get_dgl_version():
    """Get the DGL version if available."""
    return DGL_VERSION

__all__ = [
    'DGL_AVAILABLE',
    'DGL_VERSION', 
    'check_dgl_required',
    'get_dgl_version',
    'dgl',  # For conditional imports
]
