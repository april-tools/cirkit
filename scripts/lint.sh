#!/bin/bash

failed_linters=()

verbose_diff="" # For black and isort
verbose=""      # For pydocstyle and pylint
# No verbose for mypy -- output too much
if [[ $1 == "--verbose" ]]; then
    verbose_diff="--verbose --diff"
    verbose="--verbose"
    shift  # remove $1
fi

if [[ $# -gt 0 ]]; then
    files=$@
else
    files=$(git ls-files "*.py")
fi

RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

run_linter() {
    linter=$1
    shift  # remove $linter from args
    echo -e "${CYAN}Running $linter...${NC}"

    $linter $@ $files
    if [[ $? -ne 0 ]]; then
        failed_linters+=("$linter")
    fi

    echo
}

run_linter "black" "$verbose_diff" --check
run_linter "isort" "$verbose_diff" --check
run_linter "pydocstyle" "$verbose"
run_linter "pylint" "$verbose" --persistent=n
run_linter "mypy"

if [[ ${#failed_linters[@]} -gt 0 ]]; then
    echo -e "${RED}Failed: ${failed_linters[@]}${NC}"
    exit 1
fi
