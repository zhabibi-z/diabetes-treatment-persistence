# Copyright (c) 2026 Zia Habibi
# SPDX-License-Identifier: MIT
# Streamlit Cloud entry-point shim.
# The canonical app lives at src/app/app.py — this file executes it so that
# Streamlit Cloud settings (hardcoded to streamlit_app/app.py) keep working
# without a settings change.
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
_main = _root / "src" / "app" / "app.py"

exec(  # noqa: S102
    compile(_main.read_text(), str(_main), "exec"),
    {"__file__": str(_main), "__name__": "__main__"},
)
