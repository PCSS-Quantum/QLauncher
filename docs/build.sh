#!/bin/bash

# Builds the documentation 

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

mkdir -p tutorials
cp ./../examples/*.ipynb ./tutorials

make clean
make html