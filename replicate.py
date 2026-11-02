"""Master entry point: python replicate.py all [--stata /path/to/stata]."""
import os
import sys
from pathlib import Path

# Reproducible single-process numerical execution; no application-wide changes.
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(name,'1')
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent/'code'))
from reproduce import main

if __name__=='__main__':
    main()
