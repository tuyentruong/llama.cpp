#!/usr/bin/env python
"""
Runs llama.cpp in a subprocess and returns the output.
"""
import argparse
from enum import Enum, auto
import os
import subprocess

import numpy as np


class Action(Enum):
    GENERATE_EMBEDDINGS = auto()


def main(pipeline):
    print(f'running pipeline: {[action.name for action in pipeline]}')
    for action in pipeline:
        print(f'running action: {action.name}')
        if action == Action.GENERATE_EMBEDDINGS:
            print('generating embeddings')
            output = subprocess.check_output(['./embedding', '-m', '../data/llama/7B/ggml-model-q4_0.bin', '-s', '1234', '-p', 'Account - Delete - Done'])
            embedding_values = output.split()
            embedding = np.array([np.float32(i) for i in embedding_values]).reshape(1, -1)
            print(embedding.shape)
            print(embedding)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, required=False, help='action to perform')
    opt = parser.parse_args()

    if opt.action is not None:
        pipeline = [Action[opt.action]]
    else:
        # DIRECTION: change this to modify the action taken
        pipeline = [Action.GENERATE_EMBEDDINGS]
    main(pipeline)
