import argparse
from fileio import load_data
import multiprocessing
import time
import threading
from sklearn.model_selection import train_test_split
from classifier import TicTacToeClassifier, classifiers
from tic_tac_toe import TicTacToe

parser = argparse.ArgumentParser(description='test perceptron classifiers.')
parser.add_argument('--boards', metavar='FILE', type=str, required=True,
                    help='filename to read sample data from, with columns: a1,a2,..,c3,label')

parser.add_argument('--classifier', metavar='C', type=str, default='perceptron',
                    help='one of: perceptron|svm|dkp|fcnn defaults to perceptron')

parser.add_argument('--test', metavar='YN', type=float, default=0.2,
                    help='percent of all boards to reserve for test (as n: 0-1)')

parser.add_argument('--train', metavar='XN', type=float, default=1,
                    help='percent of all boards to train with (n 0-1, after test)')

parser.add_argument('--sample-runs', metavar='N', type=int, default=1,
                    help='number of runs to compute accuracy')

parser.add_argument('--multiprocessing', action='store_true',
                    help='try to consume multiple threads')

parser.add_argument('--verbose', action='store_true',
                    help='show extra details eg multiprocessing or failed matches (only on 1 sample run)')

args = parser.parse_args()


def run(miss, step):
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
    elif not args.multiprocessing:
        print('.', end='', flush=True)

    y_hat = c.test(X_test)

    index = 0
    for board, label in list(zip(X_test, y_test)):
        if y_hat[index] != label:
            miss += 1
            if args.verbose and args.sample_runs == 1:
                print(TicTacToe.pretty_print(board), 'expected:',
                      label, 'received:', y_hat[index])

        index += 1
    return miss


def report_metrics(misses):
    print("\n{:.3f}% success".format(
        100 * (1 - misses/(num_test * args.sample_runs))))
    print(time.time() - start, 'seconds elapsed')


def run_threaded(data_q, task_q, thread_index):
    step = thread_index
    while True:
        tasks = task_q.get()
        if len(tasks) > 0:
            task = tasks.pop()
            task_q.put(tasks)

            thread_miss = task(0, step)
            step += 1
            data_q.put((thread_index, step, thread_miss))
        else:
            task_q.put(tasks)
            break
    task_q.put(tasks)


def manage_data(data_q, misses, task_count=0):
    tally = 0
    while True:
        thread_index, step, data = data_q.get()
        if args.verbose:
            print('[manage_data] - completed run',
                  step, 'for thread', thread_index, flush=True)

        misses += data
        tally += 1

        if tally >= task_count:
            report_metrics(misses)
            break


start = time.time()

test = max(min(args.test, 1), 0)
train = max(min(args.train, 1), 0)
train = 1/train if train > 0 else 1
print('preparing data')

X_raw, y_raw = load_data(args.boards)
num_test = test * len(y_raw)

print('total boards available', len(y_raw))

thread_count = min(multiprocessing.cpu_count() -
                   1 if args.multiprocessing else 1, args.sample_runs)
if args.multiprocessing:
    print('starting {} threads'.format(thread_count))

    data_q = multiprocessing.Queue()
    task_q = multiprocessing.Queue()

misses = 0

if thread_count > 1:
    tasks = []
    threads = []
    threading.Thread(target=manage_data, args=(
        data_q, misses, args.sample_runs)).start()

for step in range(args.sample_runs):
    if thread_count > 1:
        tasks.append(run)
    else:
        misses = run(misses, step)

if thread_count > 1:
    pool = multiprocessing.Pool(processes=thread_count)
    for index in range(thread_count):
        p = pool.Process(target=run_threaded, args=(
            data_q, task_q, index), group=None)
        p.start()
        threads.append(p)
    task_q.put(tasks)
    pool.close()
    pool.join()
else:
    report_metrics(misses)
