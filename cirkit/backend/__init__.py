"""
This module contains code that makes the symbolic representations concrete - i.e. it translates the
symbolic computational graph to a computational graph with parameters that supports
back-propagation. The backend decides data type precision, folding and how parameters are allocated
in the computational graph. We currently support PyTorch as a backend.
"""
