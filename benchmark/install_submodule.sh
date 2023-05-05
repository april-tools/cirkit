#!/bin/bash

if [[ "$@" == *"--help"* ]]; then
  echo "Usage: $0 [--help] [all] [pyjuice] [einet]"
  echo "  --help    Print this help message"
  echo "  all       Install all the following"
  echo "  pyjuice   Install submodule for pyjuice"
  echo "  einet     Install submodule for einet"
  print_help
  exit 0
fi

if [[ "$@" == *"pyjuice"* || "$@" == *"all"* ]]; then
  echo "Installing submodule for pyjuice: "$(dirname $0)/pyjuice/pyjuice""
  pip install --no-deps -e "$(dirname $0)/pyjuice/pyjuice"
fi

if [[ "$@" == *"einet"* || "$@" == *"all"* ]]; then
  echo "NOT IMPLEMENTED!"
  # TODO: add commands for einet
fi
