"""PostToolUse hook: format and lint-fix a Python file right after Claude edits it.

Reads the hook's JSON from stdin, so it works the same on Windows, macOS and Linux.
Never blocks Claude: if ruff is missing or the file isn't Python, it does nothing.
"""

import json
import shutil
import subprocess
import sys

try:
    data = json.load(sys.stdin)
except ValueError:
    sys.exit(0)
path = (data.get("tool_input") or {}).get("file_path", "")
ruff = shutil.which("ruff")
if path.endswith(".py") and ruff:
    subprocess.run([ruff, "check", "--fix", "--quiet", path], check=False)
    subprocess.run([ruff, "format", "--quiet", path], check=False)
sys.exit(0)
