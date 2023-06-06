#!/usr/bin/bash

if [[ ! "$*" =~ (all|pyjuice|einet) ]]; then
  if [[ "$@" != *"--help"*  ]]; then
    echo "Wrong usage!"
    echo
  fi
  echo "Usage: $0 [--help] [all] [pyjuice] [einet]"
  echo "  --help    Print this help message"
  echo "  all       Install all the following"
  echo "  pyjuice   Install submodule for pyjuice"
  echo "  einet     Install submodule for einet"
  exit 0
fi

# juice.jl does not include pip dependency

if [[ "$@" == *"pyjuice"* || "$@" == *"all"* ]]; then
  echo "Installing submodule for pyjuice: "$(dirname $0)/pyjuice/pyjuice-april""
  pip install -e "$(dirname $0)/pyjuice/pyjuice-april"
fi

if [[ "$@" == *"einet"* || "$@" == *"all"* ]]; then
  echo "NOT IMPLEMENTED!"
  # TODO: add commands for einet
fi
