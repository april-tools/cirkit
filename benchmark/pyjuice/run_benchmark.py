import os
import subprocess

env = os.environ.copy()

command = ["python", "run_pyjuice.py", "--mode", "eval"]
env["JUICE_COMPILE_FLAG"] = str(0b0011_1111_1111_1111)

result = subprocess.run(
    command, capture_output=True, cwd=os.path.split(__file__)[0], check=False, text=True, env=env
)
print("STDERR:")
print(result.stderr)  # have an extra newline
result.check_returncode()

output = result.stdout.splitlines()
print("STDOUT:")
print(output[0], output[1], output[-1], sep="\n")

results = [{"time": ln.split()[0], "mem": ln.split()[1]} for ln in output[2:-1]]
print(results)
