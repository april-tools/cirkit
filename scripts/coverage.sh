#!/bin/bash

set -e

# Run coverage and print text report
coverage run -m pytest
coverage report

# If required, generate report in additional format
if [[ $# -ge 1 ]]; then
    coverage "$1"
    shift
fi
