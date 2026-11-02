"""Raw-source reconstruction entry point, separate from prepared-data replication.

No existing prepared input or immutable reference is overwritten. See
docs/RAW_TO_RESULTS.md for the remaining raw-to-results completion requirements.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / 'code'))
from raw_sources import main

if __name__ == '__main__':
    main()
