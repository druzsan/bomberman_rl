"""The vendored `lib/` copies must match the source.

Agents import `from .lib import ...`, so a change to `lib/` at the repository
root has no effect on any agent until `tools.sync_lib` copies it.  Nothing in
the normal edit-and-test loop notices: the tests import the root `lib`, pass,
and the agent keeps running the old code.

This is not hypothetical.  It cost a four-arm, 2 000-round evaluation batch in
E17 -- every arm failed on `ImportError: cannot import name 'load_all'`, because
the function existed at the root and not in the copy the agent actually loads.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


class VendoredLibTest(unittest.TestCase):
    def test_every_vendored_copy_is_current(self):
        r = subprocess.run([sys.executable, "-m", "tools.sync_lib", "--check"],
                           cwd=REPO, capture_output=True, text=True, check=False)
        self.assertEqual(r.returncode, 0,
                         f"{r.stdout.strip()}{r.stderr.strip()}\n"
                         "run: uv run python -m tools.sync_lib")


if __name__ == "__main__":
    unittest.main()
