from copy import deepcopy
from itertools import product
from math import e, pi
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np


def read_fasta(input_fasta: str, N: int = 50000) -> Iterator[Tuple[str, str]]:
    """
    Yields header and sequences in fasta file
    Support for multi-line and single_line formats

    By default, it will only return the first 50000 sequences
    """
    N_seqs = 0
    with open(input_fasta, "r") as inpt:
        header = None

        for line in inpt:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(sequence).upper()
                    N_seqs += 1
                    if N_seqs >= N:
                        break
                header = line
                sequence = []
            else:
                sequence.append(line)

        if header is not None:
            yield header, "".join(sequence).upper()


def weight_matrix(dict_aa: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    2 out of 7 residues are core positions, so the the weight of the other 5 has to be
    distributed in these two, thus 5/2=2.5
    """
    weighted_aa = deepcopy(dict_aa)

    for aa in weighted_aa.keys():
        for register in ["a", "d"]:
            weighted_aa[aa][register] **= 2.5

    return weighted_aa


def compute_g(score: float, mean: float, sd: float) -> float:
    """
    Computes a g score from  the normal distribution
    Not to be confused with COILS score, stored in the full_coils_matrix
    """
    g = (sd * 2 * pi) ** (-1) * e ** (-0.5 * ((score - mean) / sd) ** 2)
    return g


def compute_geo_mean(vec_win_score: np.ndarray, root_correction: float) -> np.float16:
    """
    Calculates the geometrical mean of a vector of scores.
    It depends on the root_correction attribute
    """
    # TODO: There is probably a way to implement this on the whole matrix, instead of
    # in a window-by-window basis (profiling needed)
    return np.prod(vec_win_score) ** (1.0 / (root_correction))


class PyCOILS:
    def __init__(self):
        self.register = "abcdefg"
        self.register_size = len(self.register)
        self.source_path = Path(__file__).resolve().parent

        dict_ref_params = {}
        dict_mat_filename = {"MTK": "old.mat", "MTIDK": "new.mat"}

        # process parameters for both matrix types
        for mat_name, mat_file in dict_mat_filename.items():
            dict_stats, dict_aa = self._load_param_file(
                f"{self.source_path}/matrices/{mat_file}"
            )

            # process parameters for unweighted and weighted predictions
            for weighted in ["uw", "w"]:

                # process parameters for each window size
                for window_size in [14, 21, 28]:
                    print(f"Loading parameters for {mat_name}/{weighted}/{window_size}")

                    # compute the weighting and the root correction if necessary
                    if weighted == "w":
                        dict_mat = weight_matrix(dict_aa)
                        n_cores = window_size / 3.5
                        root_correction = n_cores * 2.5 + (window_size - n_cores)

                    else:
                        dict_mat = dict_aa
                        root_correction = window_size

                    # point towards the specific stats=[cc_m, cc_sd, g_m, g_sd, sc] and matrix
                    spec_stats = dict_stats[weighted][str(window_size)]
                    spec_mat = dict_mat

                    # save the specific combination of stats, matrix, and root_correction
                    spec_id = f"{mat_name}/{weighted}/{window_size}"
                    dict_ref_params[spec_id] = {}

                    dict_ref_params[spec_id]["stats"] = spec_stats
                    dict_ref_params[spec_id]["mat"] = spec_mat
                    dict_ref_params[spec_id]["root_corr"] = root_correction

        self.dict_ref_params = dict_ref_params

    def predict_sequence(
        self,
        sequence: str,
        window_size: int = 21,
        matrix: str = "MTIDK",
        weighted: str = "w",
        frame: str = "best",
        return_scores: bool = False,
    ) -> Dict[str, np.ndarray]:

        # sanitize input parameters
        assert self._sanitize_input(
            sequence, window_size, matrix, weighted, frame, return_scores
        )

        # compute the full matrix scores
        full_coils_matrix = self._compute_full_matrix(
            sequence, window_size, matrix, weighted
        )

        # set parameters for COILS computation
        spec_id = f"{matrix}/{weighted}/{window_size}"
        dict_spec_params = self.dict_ref_params[spec_id]
        spec_stats = dict_spec_params["stats"]
        cc_m, cc_sd, g_m, g_sd, sc = spec_stats

        # compute probabilities from scores
        if frame == "best":
            # perform maximization routine -> find the best score for a given residue
            # across *register* and *window_size*.
            final_scores_mat = full_coils_matrix.max(axis=0).max(axis=1)

        # after input sanitization, this must be 'allframe'
        else:
            # perform a partial maximization routine -> find the best score for a given
            # residue/register
            final_scores_mat = full_coils_matrix.max(axis=2)

        # compute COILS probabilities; if only the scores are needed, this can be suppressed
        G_cc = compute_g(final_scores_mat, cc_m, cc_sd)
        G_g = compute_g(final_scores_mat, g_m, g_sd)
        coils_prob = G_cc / (sc * G_g + G_cc)

        result = {}

        result["coils_prob"] = coils_prob

        if return_scores == True:
            result["coils_scores"] = final_scores_mat

        return result

    def run_on_file(
        self,
        input_fasta: str,
        window_size: int = 21,
        matrix: str = "MTIDK",
        weighted: str = "w",
        frame: str = "best",
        return_scores: bool = False,
    ) -> Iterator[Tuple[str, Dict[str, np.ndarray]]]:

        generator_sequences = read_fasta(input_fasta)
        for header, sequence in generator_sequences:
            results = self.predict_sequence(
                sequence, window_size, matrix, weighted, frame, return_scores
            )

            yield header, results

    def test_vs_COILS(self, graphics: bool = False) -> None:
        """
        This function checks the consistency of this COILS implementation with the GCN4
        prediction of the original COILS, for [u/uw], MTK, [14,21,28] parameters
        """
        # Test loading a sequence
        test_sequence = f"{self.source_path}/tests/gcn4.fasta"
        seq_reader = read_fasta(test_sequence)
        _, gcn4_sequence = next(seq_reader)

        list_test_params = list(product(["w", "uw"], ["MTK"]))
        list_win_size = [14, 21, 28]

        pycoils = PyCOILS()

        for weight, matrix in list_test_params:
            # load reference prediction from the original COILS
            ref_filename = f"{self.source_path}/tests/original_coils/COILS_GCN4_{weight}_{matrix}.results"
            data_val = []
            with open(ref_filename, "r") as inpt:
                for line in inpt:
                    data = line.strip().split()
                    data_val.append([float(data[i]) for i in [4, 7, 10]])
            mat_val = np.array(data_val)

            # compute prediction with this implementation of COILS
            data_pred = []
            for i_win_size in list_win_size:
                pycoils_pred = pycoils.predict_sequence(
                    gcn4_sequence,
                    window_size=i_win_size,
                    matrix=matrix,
                    weighted=weight,
                )
                data_pred.append(pycoils_pred["coils_prob"])
            mat_pred = np.array(data_pred).T

            # compare reference vs prediction values
            for i in range(3):
                reference = mat_val[:, i]
                prediction = mat_pred[:, i]
                # print(reference[:10])
                # print(prediction[:10])
                assert np.allclose(reference, prediction, atol=5e-2)
                print(
                    f"COILS vs PyCOILS @ matrix {matrix} {weight} {list_win_size[i]}: [OK]"
                )
                if graphics:
                    # this import is here to avoid creating an unnecessary dependency when running
                    # predictions
                    import matplotlib.pyplot as plt  # type: ignore

                    plt.plot(reference, alpha=0.5)
                    plt.plot(prediction, alpha=0.5)
                    plt.show()

    def _load_param_file(
        self, mat_file: str
    ) -> Tuple[Dict[str, Dict[str, List]], Dict[str, Dict[str, float]]]:
        """
        This function loads data & parameters from old.mat and new.mat
        Each contains a matrix of substitutions (MTK or MTIDK),
        and 6 sets statistical parameters (w/uw x 14/21/28)

        returns dict_states, dict_aa
        """

        dict_stats: Dict[str, Dict[str, List]] = {}
        dict_aa: Dict[str, Dict[str, float]] = {}
        register = self.register

        with open(mat_file, "r") as inpt:
            for line in inpt:
                # this is a line with comment
                if line.startswith("%"):
                    continue

                # line containing stat parameter
                elif line[0] in ["w", "u"]:
                    # read stat params
                    data_stats = line.strip().split()
                    labels = data_stats[:2]  # w/uw, window_size
                    weighted_flag, window_size = labels

                    # store stat params
                    if weighted_flag not in dict_stats:
                        dict_stats[weighted_flag] = dict()
                    parameters = [float(i) for i in data_stats[2:]]
                    dict_stats[weighted_flag][window_size] = parameters

                # aa composition matrix line
                else:
                    # read matrix composition data
                    data_aa = line.strip().split()
                    aa = data_aa[0]
                    composition = [float(i) for i in data_aa[1:]]

                    # store matrix composition data
                    dict_aa[aa] = dict()
                    for i_register, i_score in zip(register, composition):
                        dict_aa[aa][i_register] = i_score

        return dict_stats, dict_aa

    def _sanitize_input(
        self,
        sequence: str,
        window_size: int,
        matrix: str,
        weighted: str,
        frame: str,
        return_scores: bool,
    ) -> bool:

        try:
            assert type(sequence) == str
            assert window_size in [14, 21, 28]
            assert matrix in ["MTK", "MTIDK"]
            assert weighted in ["w", "uw"]
            assert frame in ["best", "all"]
            assert return_scores in [True, False]
            return True
        except AssertionError:
            print("Your input parameters are wrong and you should be ashamed")
            return False

    # @profile
    def _compute_full_matrix(
        self, sequence: str, window_size: int, matrix: str, weighted: str
    ) -> np.ndarray:
        """
        This function takes a sequence (string, uppercase) and the parameters
        window_size: int (21), weighted: bool (True)
        And returns a full COILS score matrix, from which the final results are derived
        """
        # set contextual variables
        register = self.register
        register_size = self.register_size
        sequence_size = len(sequence)

        # set parameters for COILS computation
        spec_id = f"{matrix}/{weighted}/{window_size}"
        dict_spec_params = self.dict_ref_params[spec_id]

        spec_mat = dict_spec_params["mat"]
        root_correction = dict_spec_params["root_corr"]

        # create matrix to store the COILS scores
        mat_score_full = np.zeros(shape=(register_size, sequence_size, window_size))

        # iterate through every window in the sequence
        # (the =1 ensures coverage of the right side)
        for left_win_edge in range(sequence_size - window_size + 1):
            seq_window = sequence[left_win_edge : left_win_edge + window_size]

            # iterate through every possible starting register position
            for i_start_register, _ in enumerate(register):

                # create vector to store window values
                # TODO: This could be put higher in the loop, or maybe create the
                # vector higher up, and zero it at this place
                vec_win_score = np.zeros(shape=window_size)

                # iterate through every position in the window
                for i_local_aa, aa in enumerate(seq_window):
                    # compute the global residue index
                    global_resi_index = left_win_edge + i_local_aa

                    # compute the heptad register of the current residue
                    # this depends of where is the sequence assumed to begin
                    global_heptad_pos = (
                        i_start_register + global_resi_index
                    ) % register_size
                    global_register = register[global_heptad_pos]

                    # assign the substitution score to a specific position of the residue
                    vec_win_score[i_local_aa] = spec_mat[aa][global_register]

                # compute the score for the window
                geo_mean = compute_geo_mean(vec_win_score, root_correction)

                # store the previous score in every position of the window (is a diagonal in 3d)
                mat_score_full[
                    i_start_register,
                    left_win_edge + np.arange(window_size),
                    np.arange(window_size),
                ] = geo_mean
        return mat_score_full


if __name__ == "__main__":
    pycoils = PyCOILS()

    # Test loaded parameters
    print(pycoils.dict_ref_params.keys())
    # print(pycoils.dict_ref_params)

    # Test loading a sequence
    test_sequence = f"{pycoils.source_path}/tests/gcn4.fasta"
    seq_reader = read_fasta(test_sequence)
    header, sequence = next(seq_reader)

    # Test running a prediction
    results = pycoils.predict_sequence(sequence)
    coils_prob = results["coils_prob"]

    print(coils_prob.shape)
    # print(coils_prob)

    pycoils.test_vs_COILS(graphics=False)
