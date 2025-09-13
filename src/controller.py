# Run a bunch of experiments in parallel

import subprocess
from subprocess import PIPE
import multiprocessing
from multiprocessing import cpu_count, Pool
import sys
import os

from results_tracker import ResultsTracker

# Prevent thrashing on Linux:
# Without this, each subprocess will try to use all cores
# and it's a huge performance hit.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

def test_smoothing():
    # Preliminary test, to determine best label smoothing parameter
    smoothings = [0.0, 0.1, 0.2, 0.5]
    processes = []
    for smoothing in smoothings:
        processes.append(
            subprocess.Popen([sys.executable, "train.py", "--label_smoothing", str(smoothing), "--log_file", "log" + str(smoothing)])
        )
    
    for process in processes:
        process.communicate()   # await process completion

def test_lr():
    # Preliminary test to determine best learning rate
    lrs = [0.1, 0.2, 0.5, 1.0]
    processes = []
    for lr in lrs:
        processes.append(
            subprocess.Popen([sys.executable, "train.py", "--lr", str(lr), "--log_file", "lr_" + str(lr)])
        )

    for process in processes:
        process.communicate()   # await process completion
        
def experiment():
    num_words_options = [16, 64, 256, 1024]

    embed_dim_options = [2, 4, 6, 8, 12, 16, 24, 32]
    attn_layers_options = [1, 2, 4, 6, 10]
    commands = []

    rt = ResultsTracker(num_words_options)

    counter = 0
    for num_words in num_words_options:
        for embed_dim in embed_dim_options:
            for attn_layers in attn_layers_options:
                commands.append([sys.executable, "train.py", "--embed_dim", str(embed_dim),
                                      "--attn_layers", str(attn_layers), "--num_words", str(num_words),
                                      "--log_file", "log" + str(counter)])
                counter += 1

    # hack: only start 8 processes at once
    # to prevent resource exhaustion
    n_proc = 8
    n_blocks = (len(commands) + n_proc - 1) // n_proc
    for block_idx in range(n_blocks):
        processes = []
        # Print status updates to stderr: if the output is redirected to log file,
        # status updates should still print to screen
        print(f"Beginning block {block_idx} of processes.", file=sys.stderr)
        for idx in range(n_proc * block_idx, min(n_proc * (block_idx + 1), len(commands))):
            processes.append(subprocess.Popen(commands[idx], stdout=PIPE))
        for process in processes:
            stdout, _ = process.communicate()
            rt.parse_and_record(stdout)
        print(f"Completed block {block_idx} of processes.", file=sys.stderr)
    rt.print()

experiment()
