#!/usr/bin/env python3

import unittest
import sys, os, io

from more_itertools import collapse, distribute, nth
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from tqdm import tqdm

def run_test(test):
    testout = io.StringIO()
    stdout = io.StringIO()
    stderr = io.StringIO()

    runner = unittest.TextTestRunner(testout)

    with redirect_stdout(stdout), redirect_stderr(stderr):
        runner.run(test)

    return testout.getvalue(), stdout.getvalue(), stderr.getvalue()

def record_test(test, out_dir, testout, stdout, stderr):
    (out_dir / f'{test.id()}.unittest').write_text(testout)
    (out_dir / f'{test.id()}.stdout').write_text(stdout)
    (out_dir / f'{test.id()}.stderr').write_text(stderr)

out_dir = Path(sys.argv[1])
num_workers = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
worker_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
test_dir = Path(__file__).parents[1]

# Might eventually make sense to sort this list by some estimate of how long 
# each test will take, so keep all the task about the same running time.
loader = unittest.TestLoader()
all_tests = collapse(loader.discover(test_dir))
my_tests = nth(distribute(num_workers, all_tests), worker_id)

for test in tqdm(my_tests):
    out = run_test(test)
    record_test(test, out_dir, *out)

