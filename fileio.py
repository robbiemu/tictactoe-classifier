def load_data(fn):
    '''
    helper method to load training data
    The schema for the csv of these can be notated with letters and number for the row columns, like this: **a1**, **a2**, **a3**, **b1**, **b2**, **b3**, **c1**, **c2**, **c3**, **label**
    '''
    X_train = []
    y_train = []

    f = open(fn, "r")
    line_reader = f.readlines()
    for line in line_reader:
        a1, a2, a3, b1, b2, b3, c1, c2, c3, label = line.split(',')
        board = "".join([a1, a2, a3, b1, b2, b3, c1, c2, c3])
        X_train.append(board)
        y_train.append(label)

    f.close()

    return X_train, y_train
