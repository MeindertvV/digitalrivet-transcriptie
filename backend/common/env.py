"""API keys: from the environment, else (on Windows) the value saved with `setx`, else keys/<file>."""

import os
import sys

from common.paths import KEYS


def saved_env(name: str) -> str:
    """Environment variable, falling back on Windows to the value saved with `setx`
    (which only reaches programs started after it was run)."""
    value = os.environ.get(name, "")
    if not value and sys.platform == "win32":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as reg:
                value = winreg.QueryValueEx(reg, name)[0]
                os.environ[name] = value
        except OSError:
            pass
    return value


def read_key(name: str, file: str) -> str:
    """saved_env(name), else the contents of keys/<file>; "" if neither has it."""
    value = saved_env(name)
    if not value and (KEYS / file).is_file():
        value = (KEYS / file).read_text(encoding="utf-8").strip()
    return value
