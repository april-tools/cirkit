#!/bin/bash

CIRKIT_SRC="cirkit"
TESTS_SRC="tests"

if [ $# -gt 0 ]
then
    files=$@
else
    files="${CIRKIT_SRC} ${TESTS_SRC}"
fi

isort $files
black $files
