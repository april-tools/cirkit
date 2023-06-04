from typing import Optional, Union

import numpy as np

#: A random state type is either an integer seed value or a Numpy RandomState instance.
RandomState = Union[int, np.random.RandomState]


def check_random_state(random_state: Optional[RandomState] = None) -> np.random.RandomState:
    """Check a possible input random state and return it as a Numpy's RandomState object.

    Args:
        random_state (Optional[RandomState], optional): The random state to check.
           If None a new Numpy RandomState will be returned.
            If not None, it can be either a seed integer or a np.random.RandomState instance.
            In the latter case, itself will be returned.

    Return:
        A Numpy's RandomState object.

    Raises:
        ValueError: If the random state is not None or a seed integer or a Numpy RandomState object.
    """
    if random_state is None:
        return np.random.RandomState()
    if isinstance(random_state, int):
        return np.random.RandomState(random_state)
    if isinstance(random_state, np.random.RandomState):
        return random_state
    raise ValueError(
        "The random state must be either None, a seed integer or a Numpy RandomState object"
    )
