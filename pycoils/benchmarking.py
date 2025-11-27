import csv
import time
from pathlib import Path

from . import __version__
from .pycoils import PyCOILS, read_fasta

# benchmark parameters
list_N_seqs = [1, 10, 100, 1000]  # number of sequences to be benchmarked
N_reps = 5  # repeats for each time measurement
seq_length = 500  # target length of synthetic sequence

predictor = PyCOILS()

# obtain a sample sequence
SOURCE_PATH = Path(__file__).parent
path_test_sequence = SOURCE_PATH / "tests/gcn4.fasta"
seq_reader = read_fasta(str(path_test_sequence))
_, sequence = next(seq_reader)

# make the sequence 500 residues long
extended_sequence = sequence * (seq_length // len(sequence) + 1)
full_sequence = extended_sequence[:seq_length]

# run benchmark
list_measurements = []

for i_rep in range(N_reps):
    round = i_rep + 1
    print(f"\rRunning round {round}/{N_reps} of benchmark")
    for n_seqs in list_N_seqs:
        start_time = time.time()
        for j_seq in range(n_seqs):
            predictor.predict_sequence(full_sequence)
        end_time = time.time()
        elapsed = end_time - start_time

        print(f"Round {round}, {n_seqs} sequences: {elapsed:.6f} seconds")
        measurement = [round, n_seqs, elapsed]
        list_measurements.append(measurement)

# write measurements to file
tag = __version__
output_file = Path(f"./pycoils_{tag}_benchmark.csv")
with open(output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["round", "N_seqs", "time_seconds"])
    writer.writerows(list_measurements)
