"""MetaCut local AIGC service.

The package deliberately imports no ML framework at module import time. Model
backends must lazy-load their own optional dependencies only when selected.
"""

__version__ = "0.1.0"

