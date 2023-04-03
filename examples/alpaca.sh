#!/bin/bash

#
# Temporary script - will be removed in the future
#

cd `dirname $0`
cd ..

./main -m ../data/llama/ggml-alpaca-7b-q4.bin --color -f ./prompts/alpaca.txt -ins -b 256 --top_k 10000 --temp 0.2 --repeat_penalty 1 -t 7
