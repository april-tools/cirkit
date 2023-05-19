import os
import subprocess
from typing import List, Tuple

COMPILE_FLAG = 0b0011_1111_1111_1111
BASE_SEED = 0xC699345C

HERE = os.path.split(__file__)[0]
RESULT_DIR = os.path.join(HERE, "benchmark_results")
if not os.path.exists(RESULT_DIR):
    os.mkdir(RESULT_DIR)

command = [
    "python",
    "run_pyjuice.py",
    "--mode",
    "eval",
    "--seed",
    str(BASE_SEED),
    "--num_latents",
    "32",
    "--first_pass_only",
]
position = {"mode": 3, "seed": 5, "num_latents": 7, "first_pass_only": 8}
env = os.environ.copy()
env["JUICE_COMPILE_FLAG"] = ""


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
    result.check_returncode()

    output = result.stdout.splitlines()
    # print("STDOUT:")
    # print(output[0], output[1], output[-1], sep="\n")
    return [(float(ln.split()[0]), float(ln.split()[1])) for ln in output[2:-1]]


for compiled in (True, False):
    env["JUICE_COMPILE_FLAG"] = str(compiled * COMPILE_FLAG)
    for mode in ("batch_em", "full_em", "eval"):
        command[position["mode"]] = mode
        for num_latents in (16, 32, 64):
            command[position["num_latents"]] = str(num_latents)
            for first_pass_only in (False, True):
                command = command[: position["first_pass_only"]] + (
                    ["--first_pass_only"] if first_pass_only else []
                )
                # add a dummy entry so that the first is always ignored
                results: List[Tuple[float, float]] = [(0, 0)] if first_pass_only else []
                for seed in range(1, 10) if first_pass_only else (0,):
                    command[position["seed"]] = str(BASE_SEED + seed)
                    results.extend(parse_run())

                with open(
                    os.path.join(
                        RESULT_DIR, f"{compiled=}-{mode=}-{num_latents=}-{first_pass_only=}.csv"
                    ),
                    "w",
                    encoding="utf-8",
                ) as f:
                    for t, m in results[1:]:
                        print(f"{t},{m}", file=f)
