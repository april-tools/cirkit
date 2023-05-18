import os
import subprocess
from typing import cast

env = os.environ.copy()

command = ["python", "run_pyjuice.py", "--mode", "eval"]
env["JUICE_COMPILE_FLAG"] = str(0b0011_1111_1111_1111)

try:
    result = subprocess.run(
        command, capture_output=True, cwd=os.path.split(__file__)[0], check=True, text=True, env=env
    )
except subprocess.CalledProcessError as e:
    print(cast(str, e.stderr))  # should be str with text=True
    raise e

print("STDERR:")
print(result.stderr)  # have an extra newline
print("STDOUT:")
print(result.stdout, end="")
