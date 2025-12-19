"""Entry point for VideoTuner when run as a module or compiled executable."""

import sys

from videotuner.pipeline import main

if __name__ == "__main__":
    sys.exit(main())
