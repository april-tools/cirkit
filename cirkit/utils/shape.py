def comp_shape(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
    """Checks whether shape a is compatible with shape b.
    Two shapes are equal if all their elements match. An
    element is allowed to be left unspecified (using -1)
    which is considered as matching element.
    Moreover, one of the two shapes is allowed to be composed
    of an additional element -1 on posision 0, in which case
    they are still considered as compatible shapes.

    Args:
        a (tuple[int, ...]): Shape a
        b (tuple[int, ...]): Shape b

    Returns:
        bool: Whether the two shapes are compatible.
    """
    # TODO: Can be generalized to allow arbitrary number of -1
    # while still being compatible if everything besides -1s matches
    # since we can have a routine that unsqueezes needed dimensions
    # and expands them as needed.
    # TODO: given a more mature support of named tensors
    # https://pytorch.org/docs/stable/named_tensor.html
    # it could be possible to drop this function and rely on them
    num_dim = min(len(a), len(b))

    for a_i, b_i in zip(a[::-1], b[::-1]):
        if (a_i != -1) and (b_i != -1) and a_i != b_i:
            return False

    return (
        len(a) == len(b)
        or (len(a) == (len(b) + 1) and a[0] == -1)
        or (len(a) + 1 == len(b) and b[0] == -1)
    )
