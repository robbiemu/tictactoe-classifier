import numpy as np
from random import random
from tic_tac_toe import TicTacToe


class TicTacToeGenerator:
    '''
    various methods of generating plausable tic tac toe boards
    '''

    def generate_naive(self):
        tiles = [-1, 0, 1]
        return np.random.choice(tiles, 9)

    def generate_weighted(self):
        tiles = [-1, 0, 1]

        p_blank = random() / 3 + 0.5
        p_tile = (1 - p_blank) / 2

        return np.random.choice(tiles, 9, p=[p_tile, p_blank, p_tile])

    def generate_valid(self):
        tile = [1, -1]
        turn = 0
        board = np.zeros(9)
        while(np.any(board == 0)):
            index = np.random.choice(np.nonzero(board == 0)[0], 1)
            board[index] = tile[turn % 2]
            turn += 1
            if TicTacToe.is_solved(board) or random() > 0.75:
                break
        return board
