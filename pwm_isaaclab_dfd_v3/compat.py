from __future__ import annotations

import sys


try:
    import colorama as colorama
except ImportError:
    class _EmptyColors:
        BLACK = ""
        BLUE = ""
        CYAN = ""
        GREEN = ""
        MAGENTA = ""
        RED = ""
        RESET = ""
        RESET_ALL = ""
        WHITE = ""
        YELLOW = ""

    class _ColoramaFallback:
        Fore = _EmptyColors()
        Back = _EmptyColors()
        Style = _EmptyColors()

    colorama = _ColoramaFallback()
    sys.modules.setdefault("colorama", colorama)
