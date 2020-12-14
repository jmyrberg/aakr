#!/bin/sh

conda activate aakr && \
rm -rf docs/build && \
sphinx-apidoc -f -o docs/source aakr && \
rm -rf docs/source/modules.rst && \
sphinx-build -b html docs/source docs/build
open -a /Applications/Safari.app file:///Users/e103089/Documents/Personal/aakr/docs/build/index.html
