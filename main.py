#!/usr/bin/env python3
import os
import pickle
import argparse
import logging

from examples import get_example
from lr_attack import run


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('model', help='name of the model to attack')
    parser.add_argument('--image', type=int, default=0)
    parser.add_argument('--accuracy', action='store_true', help='first determines the accuracy of the model')
    parser.add_argument('--save', type=str, default=None, help='filename to save result to')

    # hyperparameters
    parser.add_argument('--regions', type=int, default=400)
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--gamma', type=int, default=6, help='hyperparam of region selection')
    parser.add_argument('--misc-factor', type=float, default=75.)

    # advanced control over certain aspects (only if you know what you are doing)
    parser.add_argument('--nth-likely-class-starting-point', type=int, default=None)
    parser.add_argument('--no-line-search', action='store_true')
    parser.add_argument('--max-other-classes', type=int, default=None)
    parser.add_argument('--no-normalization', action='store_true')

    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    if args.save is not None:
        if os.path.exists(args.save):
            logging.warning(f'not runnning because results already exist: {args.save}')
            return

    result = run(*get_example(args.model), args=args)

    if args.save is not None:
        directory = os.path.dirname(args.save)
        if len(directory) > 0 and not os.path.exists(directory):
            os.makedirs(directory)
        with open(args.save, 'wb') as f:
            pickle.dump(result, f)


if __name__ == '__main__':
    main()
