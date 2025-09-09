import os
import sys

import nfeloqb

arg = sys.argv[1] if len(sys.argv) > 1 else "run"

# Default local runs to disable Airtable
os.environ.setdefault("NFELOQB_DISABLE_AIRTABLE", "1")

if arg == "run":
    nfeloqb.run()
elif arg == "run_now":
    nfeloqb.run(force_run=True)
else:
    print("Usage: python workflow.py [run|run_now]")
