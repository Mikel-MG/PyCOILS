"""
Test 1: Sequence parser
Test 2: Compare vs original COILS
Test 3: GCN4 prediction
"""

from itertools import product
from pathlib import Path

import numpy as np

from .pycoils import PyCOILS, read_fasta

SOURCE_PATH = Path(__file__).parent
path_test_sequence = SOURCE_PATH / "tests/gcn4.fasta"


def test_sequence_parser() -> None:
    """
    Check whether a FASTA file is parsed correctly
    """
    seq_reader = read_fasta(str(path_test_sequence))
    header, sequence = next(seq_reader)
    expected_header = ">sp|P03069.1|GCN4_YEAST RecName: Full=General control protein GCN4, AltName: Full=Amino acid biosynthesis regulatory protein"
    expected_sequence = "MSEYQPSLFALNPMGFSPLDGSKSTNENVSASTSTAKPMVGQLIFDKFIKTEEDPIIKQDTPSNLDFDFALPQTATAPDAKTVLPIPELDDAVVESFFSSSTDSTPMFEYENLEDNSKEWTSLFDNDIPVTTDDVSLADKAIESTEEVSLVPSNLEVSTTSFLPTPVLEDAKLTQTRKVKKPNSVVKKSHHVGKDDESRLDHLGVVAYNRKQRSIPLSPIVPESSDPAALKRARNTEAARRSRARKLQRMKQLEDKVEELLSKNYHLENEVARLKKLVGER"
    assert header == expected_header
    assert sequence == expected_sequence


def test_vs_COILS(graphics) -> None:
    """
    This function checks the consistency of this COILS implementation with the GCN4
    prediction of the original COILS, for [u/uw], MTK, [14,21,28] parameters
    """
    predictor = PyCOILS()

    seq_reader = read_fasta(str(path_test_sequence))
    _, gcn4_sequence = next(seq_reader)

    list_test_params = list(product(["w", "uw"], ["MTK"]))
    list_win_size = [14, 21, 28]

    for weight, matrix in list_test_params:
        # load reference prediction from the original COILS
        ref_filename = (
            SOURCE_PATH / f"tests/original_coils/COILS_GCN4_{weight}_{matrix}.results"
        )
        data_val = []
        with open(ref_filename, "r") as inpt:
            for line in inpt:
                data = line.strip().split()
                data_val.append([float(data[i]) for i in [4, 7, 10]])
        mat_val = np.array(data_val)

        # compute prediction with this implementation of COILS
        data_pred = []
        for i_win_size in list_win_size:
            pycoils_pred = predictor.predict_sequence(
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
                # this import is here to avoid creating an unnecessary dependency
                # when running predictions
                import matplotlib.pyplot as plt  # type: ignore

                plt.plot(reference, alpha=0.5)
                plt.plot(prediction, alpha=0.5)
                plt.show()


def test_prediction():
    """
    Check whether prediction of protein sequence (GCN4) is consistent
    """
    predictor = PyCOILS()

    seq_reader = read_fasta(str(path_test_sequence))
    _, sequence = next(seq_reader)
    results = predictor.predict_sequence(
        sequence, return_scores=True, return_register=True
    )
    coils_prob = results["coils_prob"]
    coils_scores = results["coils_scores"]
    coils_register = results["coils_register"]

    print(f"Size of coiled-coil probability vector: {coils_prob.shape}")
    print(f"Size of coiled-coil score vector: {coils_scores.shape}")
    print(f"Size of register list: {len(coils_register)}")
    print()

    part_cc_prediction = "".join(["C" if i > 0.5 else "_" for i in coils_prob])[-80:]
    part_cc_register = "".join(coils_register)[-80:]

    expect_part_cc_prediction = "___________________________________CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC_"
    expect_part_cc_register = "cdefgabcdefgabcdefgabcdebbfgabcefgabcdefgabcdefgabcdefgabcdefgabcdefgabcdefgabcd"

    print(part_cc_prediction)
    print(part_cc_register)

    assert part_cc_prediction == expect_part_cc_prediction
    assert part_cc_register == expect_part_cc_register

    print(sequence[-80:])


def run_tests(graphics=False):
    """
    Run the complete testing routine
    """
    separator = "#" * 80
    print(separator)
    print("\nTesting sequence parser...\n")
    test_sequence_parser()
    print(separator)
    print("\nTesting vs original COILS...\n")
    test_vs_COILS(graphics)
    print(separator)
    print("\nTesting prediction...\n")
    test_prediction()
    print(separator)


if __name__ == "__main__":
    run_tests()
