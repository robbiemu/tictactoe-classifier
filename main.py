import argparse
from fileio import load_data
from sklearn.model_selection import train_test_split
from classifier import TicTacToeClassifier, classifiers
from tic_tac_toe import TicTacToe

parser = argparse.ArgumentParser(description='test perceptron classifiers.')
parser.add_argument('--verbose', action='store_true',
                    help='show failed matches (only on 1 sample run)')

parser.add_argument('--sample-runs', metavar='N', type=int, default='1',
                    help='number of runs to compute accuracy')

parser.add_argument('--classifier', metavar='C', type=str, default='perceptron',
                    help='one of: perceptron|svm|dkp defaults to perceptron')

parser.add_argument('--boards', metavar='FILE', type=str, required=True,
                    help='filename to read sample data from, with columns: a1,a2,..,c3,label')

parser.add_argument('--test', metavar='YN', type=float, default=0.2,
                    help='percent of all boards to reserve for test (as n: 0-1)')

parser.add_argument('--train', metavar='XN', type=float, default=1,
                    help='percent of all boards to train with (n 0-1, after test)')

args = parser.parse_args()

test = max(min(args.test, 1), 0)
train = max(min(args.train, 1), 0)
train = 1/train if train > 0 else 1
print('preparing data')

X_raw, y_raw = load_data(args.boards)
print('total boards available', len(y_raw))

miss = 0

for step in range(args.sample_runs):
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw, y_raw, test_size=test)

    offset = int(-(len(X_train_raw)//train))

    X_train = [TicTacToe.deserialize(board) for board in X_train_raw[offset:]]
    y_train = [int(label) for label in y_train_raw[offset:]]

    X_test = [TicTacToe.deserialize(board) for board in X_test_raw]
    y_test = [int(label) for label in y_test_raw]

    if step == 0:
        print('training samples', len(y_train),
              'testing samples', len(y_test), flush=True)

    classifier = classifiers[args.classifier]
    if step == 0:
        print('using classifier:', classifier, flush=True)
    c = TicTacToeClassifier(classifier)
    c.train(X_train, y_train)

    if step == 0:
        print('testing', flush=True)
    else:
        print('.', end='', flush=True)

    y_hat = c.test(X_test)

    index = 0
    for board, label in list(zip(X_test, y_test)):
        if y_hat[index] != label:
            miss += 1
            if args.verbose and args.sample_run == 1:
                print(TicTacToe.pretty_print(board), 'expected:',
                      label, 'received:', y_hat[index])

        index += 1

print("\n{:.3f}% success".format(
    100 * (1 - miss/(len(y_test) * args.sample_runs))))
