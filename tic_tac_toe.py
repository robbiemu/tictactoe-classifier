import numpy as np


class TicTacToe:
    Tiles = dict([(-1, 'O'), (0, ' '), (1, 'X')])
    TilesReverse = dict([('O', -1), (' ', 0), ('X', 1)])
    Validities = dict([(0, 'invalid'), (1, 'valid')])

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
        if (len(set([board[i][i] for i in range(len(board))])) == 1) and (board[0][0] != 0):
            return board[0][0]
        if (len(set([board[i][len(board)-i-1] for i in range(len(board))])) == 1) and (board[0][0] != 0):
            return board[0][len(board)-1]
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
        '''src https://www.geeksforgeeks.org/validity-of-a-given-tic-tac-toe-board-configuration/'''

        arr = TicTacToe.serialize(board)
        # Count number of 'X' and 'O' in the given board
        xcount = arr.count('X')
        ocount = arr.count('O')

        # Board can be valid only if either xcount and ocount
        # is same or xount is one more than oCount
        if(xcount == ocount+1 or xcount == ocount):
            # Check if O wins
            if TicTacToe.win_check(board, -1):
                # Check if X wins, At a given point only one can win,
                # if X also wins then return Invalid
                if TicTacToe.win_check(board, 1):
                    return 0

                # O can only win if xcount == ocount in case where whole
                # board has values in each position.
                if xcount == ocount:
                    return 1

            # If X wins then it should be xc == oc + 1,
            # If not return Invalid
            if TicTacToe.win_check(board, 1) and xcount != ocount+1:
                return 0

            # if O is not the winner return Valid
            if not TicTacToe.win_check(board, -1):
                return 1

        # If nothing above matches return invalid
        return 0
