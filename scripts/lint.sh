#!/bin/bash

verbose_diff="" # For black and isort
verbose=""      # For pydocstyle and pylint
# No verbost for mypy -- output too much
if [ "$1" == "--verbose" ]; then
    verbose_diff="--verbose --diff"
    verbose="--verbose"
    shift
fi

if [[ $# -gt 0 ]]; then
    file_args=$@
else
    file_args=$(git ls-files "*.py")
fi

CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}Running Black...${NC}"
black $verbose_diff --check $file_args
echo
echo -e "${CYAN}Running Isort...${NC}"
isort $verbose_diff --check $file_args
echo
echo -e "${CYAN}Running Pydocstyle...${NC}"
pydocstyle $verbose $file_args
echo
echo -e "${CYAN}Running Pylint...${NC}"
pylint $verbose --persistent=n $file_args
echo
echo -e "${CYAN}Running Mypy...${NC}"
mypy $file_args
