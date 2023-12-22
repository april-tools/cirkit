from cirkit.new.layers.inner.product import ProductLayer
from cirkit.new.layers.inner.sum import SumLayer


# NOTE: We multi-inherit so that SumProductLayer can be used at anywhere that requires a SumLayer or
#       a ProductLayer, e.g., in both SymbolicSumLayer and SymbolicProductLayer.
class SumProductLayer(SumLayer, ProductLayer):
    """The abstract base class for "fused" sum-product layers.

    The fusion of sum and product can sometimes save the instantiation of the product units, but \
    the sum units are limited to dense connection along the units dim, and the arity is only for \
    product. The sum over different layers, a MixingLayer is still required.
    """

    # NOTE: Both SumLayer and ProductLayer have no __init__, and we don't change the __init__ here,
    #       so that it falls back to InnerLayer. Although sum-prod layers typically have parameters,
    #       we still allow it to be optional for flexibility.

    # NOTE: Both SumLayer and ProductLayer have reset_parameters, so the first base, SumLayer, takes
    #       priority and properly does the reset.
