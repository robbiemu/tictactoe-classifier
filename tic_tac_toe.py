import numpy as np


class TicTacToe:
    '''
    helper class to handle tic tac toe boards
    '''

    Tiles = dict([(-1, 'O'), (0, ' '), (1, 'X')])
    TilesReverse = dict([('O', -1), (' ', 0), ('X', 1)])
    Validities = dict([(False, 'invalid'), (True, 'valid')])

    @staticmethod
    def serialize(board):
        return "".join([TicTacToe.Tiles[i] for i in board.tolist()])

    @staticmethod
    def deserialize(board):
        return np.array([TicTacToe.TilesReverse[i] for i in list(board)])

    @staticmethod
    def csv(board):
        return ",".join([TicTacToe.Tiles[i] for i in board.tolist()])

    @staticmethod
    def pretty_print(board):
        res = ""
        for row in board.reshape(3, 3):
            res += "".join([TicTacToe.Tiles[i] +
                            ' ' for i in row.tolist()]) + '\n'

        return res

    @staticmethod
    def is_solved(board):
        '''src https://stackoverflow.com/questions/39922967/python-determine-tic-tac-toe-winner#answer-39923094'''

        board = board.reshape(3, 3)
        # transposition to check rows, then columns
        for newBoard in [board, np.transpose(board)]:
            result = TicTacToe.check_rows(newBoard)
            if result != 0:
                return True
        return TicTacToe.check_diagonals(board) != 0

    @staticmethod
    def check_rows(board):
        for row in board:
            if (len(set(row)) == 1) and (row[0] != 0):
                return row[0]
        return 0

    @staticmethod
    def check_diagonals(board):
        '''
        the original had set the diagonals wrong
        '''
        if len(set([board[i][i] for i in range(len(board))])) == 1 and board[1][1] != 0 \
                or \
                len(set([board[i][len(board)-i-1] for i in range(len(board))])) == 1 and board[1][1] != 0:
            return board[1][1]
        return 0

    @staticmethod
    def win_check(board, tile):
        board = board.reshape(3, 3)
        # transposition to check rows, then columns
        for newBoard in [board, np.transpose(board)]:
            result = TicTacToe.check_rows(newBoard)
            if result == tile:
                return True
        return TicTacToe.check_diagonals(board) == tile

    @staticmethod
    def is_valid(board):
        '''
        original src https://www.geeksforgeeks.org/validity-of-a-given-tic-tac-toe-board-configuration/

        it was noticed that this algorithm produces this scenario:
            X O O 
            X O X 
            O X X 
        invalid

        this is strange, because the algorithm is otherwise almost always correct.
        '''

        arr = TicTacToe.serialize(board)
        # Count number of 'X' and 'O' in the given board
        x_count = arr.count('X')
        o_count = arr.count('O')

        x = TicTacToe.TilesReverse['X']
        o = TicTacToe.TilesReverse['O']

        # Board can be valid only if either x_count and o_count
        # is same or x_count is one more than o_count
        if x_count == o_count or x_count == o_count + 1:
            # Check if O wins
            if TicTacToe.win_check(board, o):
                # Check if X wins, At a given point only one can win,
                # if X also wins then return Invalid
                if TicTacToe.win_check(board, x):
                    return False

                # O can only win if x_count == o_count
                return x_count == o_count

            # If X wins then it should be x_count == o_count + 1,
            # If not return Invalid
            elif TicTacToe.win_check(board, x):
                return x_count == o_count + 1

            # if neither is the winner return Valid
            else:
                return True

        # count of moves are not correct, return Invalid
        return False
