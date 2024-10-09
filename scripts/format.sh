#!/bin/bash

if [ $# -gt 0 ]
then
    files=$@
else
    files=$(git ls-files "*.py")
fi

isort $files
black $files
