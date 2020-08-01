import argparse
from fileio import load_data
from classifier import TicTacToeClassifier
from tic_tac_toe import TicTacToe

parser = argparse.ArgumentParser(description='generate boards.')
# parser.add_argument('-p', '--pretty-print', action='store_true',
#                    help='pretty-print boards')

parser.add_argument('--train', '--train-csv', metavar='FILE', type=str,
                    help='filename to read sample data from, with columns: board, label')

parser.add_argument('--test', '--test-csv', metavar='FILE', type=str,
                    help='filename to read test data from, with columns: board, label')

args = parser.parse_args()

print('loading training data')
X_raw, y_raw = load_data(args.train)
X_train = [TicTacToe.deserialize(board) for board in X_raw]
y_train = [int(label) for label in y_raw]

print('loading test data')
X_raw, y_raw = load_data(args.test)
X_test = [TicTacToe.deserialize(board) for board in X_raw]
y_test = [int(label) for label in y_raw]

print('training')
c = TicTacToeClassifier()
c.train(X_train, y_train)

print('testing')
y_hat = c.test(X_test)

index = 0
miss = 0
for board, label in list(zip(X_test, y_test)):
    if y_hat[index] != label:
        miss += 1
        print(TicTacToe.pretty_print(board), 'expected:',
              label, 'received:', y_hat[index])

    index += 1

print("{:.3f}% success".format(100 * (1 - miss/len(y_test))))
