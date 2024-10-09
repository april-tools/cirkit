#!/bin/bash

failed_linters=()

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


if [ $# -eq 0 ] || [ $1 != "--tool" ]
then
    if [ $# -gt 0 ]
    then
        files=$@
    else
        files=$(git ls-files "*.py")
    fi
    run_linter "black" --check --verbose
    run_linter "isort" --check --verbose
    run_linter "pydocstyle" --verbose
    run_linter "pylint" --verbose
    run_linter "mypy"
else
    if [ -z $2 ]
    then
        echo "No liniting tool has been specified, exiting ..."
        exit 1
    fi
    tool=$2
    shift && shift
    if [[ $# -gt 0 ]]; then
        files=$@
    else
        files=$(git ls-files "*.py")
    fi
    if [ $tool == "black" ]
    then
        run_linter "black" --check --verbose
    elif [ $tool == "isort" ]
    then
        run_linter "isort" --check --verbose
    elif [ $tool == "pydocstyle" ]
    then
        run_linter "pydocstyle" --verbose
    elif [ $tool == "pylint" ]
    then
        run_linter "pylint" --verbose
    elif [ $tool == "mypy" ]
    then
        run_linter "mypy"
    else
        echo -e "Unknown linting tool: $tool"
    fi
fi

if [[ ${#failed_linters[@]} -gt 0 ]]; then
    echo -e "${RED}Failed: ${failed_linters[@]}${NC}"
    exit 1
fi
