import argparse
import itertools
import numpy as np
from tic_tac_toe import TicTacToe

parser = argparse.ArgumentParser(description='generate boards.')
parser.add_argument('-p', '--pretty-print', action='store_true',
                    help='pretty-print boards')
parser.add_argument('-c', '--csv', action='store_true',
                    help='print comma-separated boards')

parser.add_argument('--with-labels', action='store_true',
                    help='print validity labels')

args = parser.parse_args()

boards = [np.array(board) for board in list(
    itertools.product([-1, 0, 1], repeat=9))]

for board in boards:
    if args.pretty_print:
        representation = TicTacToe.pretty_print(board)
        if args.with_labels:
            representation += '\n' + \
                TicTacToe.Validities[TicTacToe.is_valid(board)]
    elif args.csv:
        representation = TicTacToe.csv(board)
        if args.with_labels:
            representation += ',' + str(1 if TicTacToe.is_valid(board) else 0)
    else:
        representation = TicTacToe.serialize(board)
        if args.with_labels:
            representation += ',' + str(1 if TicTacToe.is_valid(board) else 0)

    print(representation)
