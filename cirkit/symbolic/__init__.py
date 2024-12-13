"""
This module implements abstract symbolic representations of circuits.
These symbolic representations need compiling to some backend before they can be executed or used to perform inference with.
More specifically, the module defines operations on circuits, circuit components and the computational graph that defines the computed function.
The parameters are symbolically defined but not materialised/allocated in memory.
While data types are defined here, they are abstract (int, real, complex); the precision is decided at compile time based on the backend.
"""
