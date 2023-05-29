import os
import subprocess
from typing import List, Tuple

COMPILE_FLAG = 0b0011_1111_1111_1111
BASE_SEED = 42
NUM_BATCHES = 100
BATCH_SIZE = 128
K_SETTINGS = (16, 32, 64, 128, 256, 512)
NUM_FIRST_PASSES = 10

HERE = os.path.split(__file__)[0]
RESULT_DIR = os.path.join(HERE, "benchmark_results")
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

command = [
    "python",
    "run_pyjuice.py",
    "--mode",
    "eval",
    "--seed",
    str(BASE_SEED),
    "--num_batches",
    str(NUM_BATCHES),
    "--batch_size",
    str(BATCH_SIZE),
    "--region_graph",
    "../quad_tree_stdec_28_28.json",  # cwd is HERE
    "--num_latents",
    "32",
    "--first_pass_only",
]
position = {"mode": 3, "seed": 5, "num_latents": 13, "first_pass_only": 14}
env = os.environ.copy()
env["PYTHONHASHSEED"] = str(BASE_SEED)
env["JUICE_COMPILE_FLAG"] = "0"


def parse_run() -> List[Tuple[float, float]]:
    """Run benchmark once and parse the results.

    Require `command` and `env` gloablly set inplace.

    Returns:
        List[Tuple[float, float]]: The benchmark results in (time,mem) form.
    """
    result = subprocess.run(
        command,
        capture_output=True,
        cwd=HERE,
        check=False,
        text=True,
        env=env,
    )
    print("STDERR:")
    print(result.stderr)  # have an extra newline

    if result.returncode:
        return [(-1, -1)]  # meaning this run is broken

    print("STDOUT:")
    time_mem: List[Tuple[float, float]] = []
    for ln in result.stdout.splitlines():
        if ln[:5] == "t/m: ":
            t_m = ln[5:].split()
            time_mem.append((float(t_m[0]), float(t_m[1])))
        else:
            print(ln)
    print("\n")  # add an extra newline
    return time_mem


for compiled in (False, True):
    env["JUICE_COMPILE_FLAG"] = str(compiled * COMPILE_FLAG)
    for mode in ("batch_em", "full_em", "eval"):
        command[position["mode"]] = mode
        for num_latents in K_SETTINGS:
            command[position["num_latents"]] = str(num_latents)
            for first_pass_only in (False, True):
                command = command[: position["first_pass_only"]] + (
                    ["--first_pass_only"] if first_pass_only else []
                )
                # add a dummy entry so that the first is always ignored
                time_memory: List[Tuple[float, float]] = [(-1, -1)] if first_pass_only else []
                for seed in range(1, 10) if first_pass_only else (0,):
                    env["PYTHONHASHSEED"] = command[position["seed"]] = str(BASE_SEED + seed)
                    time_memory.extend(parse_run())

                with open(
                    os.path.join(
                        RESULT_DIR,
                        f"{compiled=}-{mode=}-batch_size={BATCH_SIZE}"
                        f"-{num_latents=}-{first_pass_only=}.csv",
                    ),
                    "w",
                    encoding="utf-8",
                ) as f:
                    for t, m in time_memory[1:]:
                        print(f"{t},{m}", file=f)
