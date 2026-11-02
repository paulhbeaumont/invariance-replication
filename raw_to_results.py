"""Clean raw-to-results entry point; preserves the prepared-data launcher."""
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(variable,'1')
sys.path.insert(0,str(Path(__file__).resolve().parent/'code'))
from full_raw_verification import main

if __name__=='__main__':
    main()
