def load_data(fn):
    X_train = []
    y_train = []

    f = open(fn, "r")
    line_reader = f.readlines()
    for line in line_reader:
        board, label = line.split(',')
        X_train.append(board)
        y_train.append(label)

    f.close()

    return X_train, y_train
