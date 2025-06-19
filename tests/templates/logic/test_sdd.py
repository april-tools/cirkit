import itertools
import tempfile

import pytest

from cirkit.templates.logic import SDD
from cirkit.pipeline import compile

def test_compile_sdd():
    # A & (~ B | C)
    sdd_s_1 = "L 1 0 1\nL 3 2 2\nL 4 4 3\nL 5 2 -2\nT 6\nD 2 3 2 3 4 5 6\nL 7 0 -1\nF 8\nD 0 1 2 1 2 7 8"

    sdd_c = SDD.from_string(sdd_s_1)
    # construct circuit without enforcing smoothness
    s_c = sdd_c.build_circuit(enforce_smoothness=False)
    t_c = compile(s_c)

    assert t_c.properties.decomposable
    assert not t_c.properties.smooth

    sdd_c = SDD.from_string(sdd_s_1)
    # construct circuit and enforce smoothness
    s_c = sdd_c.build_circuit(enforce_smoothness=True)
    t_c = compile(s_c)

    assert t_c.properties.decomposable
    assert t_c.properties.smooth
    assert t_c.properties.structured_decomposable
