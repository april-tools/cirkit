#!/bin/bash

set -e

format=""
if [[ $# -ge 1 ]] && [[ "${1:0:2}" = "--" ]]; then
    format="${1#--}"  # e.g. --xml for xml
    shift  # remove $1
fi

# Run coverage and print text report
coverage run -m pytest $@  # all the rest args
coverage report

# If required, generate report in additional format
if [[ -n $format ]]; then
    coverage $format
fi
