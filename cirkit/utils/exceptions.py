class StructuralPropertyError(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


class SymbolicOperatorNotFound(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)
