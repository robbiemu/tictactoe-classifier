import numpy as np
import argparse
from tic_tac_toe import TicTacToe
from generator import TicTacToeGenerator

parser = argparse.ArgumentParser(description='generate boards.')
parser.add_argument('-p', '--pretty-print', action='store_true',
                    help='pretty-print boards')
parser.add_argument('-c', '--csv', action='store_true',
                    help='print comma-separated boards')

parser.add_argument('--with-labels', action='store_true',
                    help='print validity labels')

parser.add_argument('--of-each', type=int,
                    help='number of each (valid/weighted/random) boards to generate')
parser.add_argument('--valid', type=int,
                    help='number of valid boards to generate')
parser.add_argument('--naive', type=int,
                    help='number of naive (fully random) boards to generate')
parser.add_argument('--weighted', type=int,
                    help='number of weighted (tending to be more empty random) boards to generate')

args = parser.parse_args()

boards = []
t = TicTacToeGenerator()

for i in range(args.naive or args.of_each or 0):
    boards.append(t.generate_naive())
for i in range(args.weighted or args.of_each or 0):
    boards.append(t.generate_weighted())
for i in range(args.valid or args.of_each or 0):
    boards.append(t.generate_valid())

for board in boards:
    if args.pretty_print:
        representation = TicTacToe.pretty_print(board)
    elif args.csv:
        representation = TicTacToe.csv(board)
    else:
        representation = TicTacToe.serialize(board)

    print("".join([representation, ',' +
                   str(TicTacToe.is_valid(board)) if args.with_labels else ""]))
