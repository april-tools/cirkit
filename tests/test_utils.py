# pylint: disable=missing-function-docstring
# TODO: disable checking for docstrings for every test file in tests/
import numpy as np

from cirkit.utils.random import check_random_state


def test_check_random_state() -> None:
    assert check_random_state().__class__ == np.random.RandomState
    assert check_random_state(42).__class__ == np.random.RandomState
    rs = np.random.RandomState(42)
    assert id(check_random_state(rs)) == id(rs)
