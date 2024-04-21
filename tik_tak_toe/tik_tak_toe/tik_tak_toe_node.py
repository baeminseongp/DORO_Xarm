#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node

from doro_interfaces.srv import Index
from math import inf as infinity
from random import choice
import platform
import time
from os import system

HUMAN = -1
COMP = +1
board = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
]


def ai_turn():

    depth = len(empty_cells(board))
    if depth == 0 or game_over(board):
        return

    if depth == 9:
        x = choice([0, 1, 2])
        y = choice([0, 1, 2])
    else:
        move = minimax(board, depth, COMP)
        x, y = move[0], move[1]
    set_move(x, y, COMP)
    print(3*x+y+1)
    return 3*x+y+1
def minimax(state, depth, player):
    if player == COMP:
        best = [-1, -1, -infinity]
    else:
        best = [-1, -1, +infinity]

    if depth == 0 or game_over(state):
        score = evaluate(state)
        return [-1, -1, score]

    for cell in empty_cells(state):
        x, y = cell[0], cell[1]
        state[x][y] = player
        score = minimax(state, depth - 1, -player)
        state[x][y] = 0
        score[0], score[1] = x, y

        if player == COMP:
            if score[2] > best[2]:
                best = score  # max value
        else:
            if score[2] < best[2]:
                best = score  # min value

    return best
def game_over(state):
    return wins(state, HUMAN) or wins(state, COMP)
def evaluate(state):
    if wins(state, COMP):
        score = +1
    elif wins(state, HUMAN):
        score = -1
    else:
        score = 0

    return score
def wins(state, player):
    win_state = [
        [state[0][0], state[0][1], state[0][2]],
        [state[1][0], state[1][1], state[1][2]],
        [state[2][0], state[2][1], state[2][2]],
        [state[0][0], state[1][0], state[2][0]],
        [state[0][1], state[1][1], state[2][1]],
        [state[0][2], state[1][2], state[2][2]],
        [state[0][0], state[1][1], state[2][2]],
        [state[2][0], state[1][1], state[0][2]],
    ]
    if [player, player, player] in win_state:
        return True
    else:
        return False
def empty_cells(state):
    cells = []

    for x, row in enumerate(state):
        for y, cell in enumerate(row):
            if cell == 0:
                cells.append([x, y])

    return cells
def valid_move(x, y):
    if [x, y] in empty_cells(board):
        return True
    else:
        return False
def set_move(x, y, player):
    if valid_move(x, y):
        board[x][y] = player
        return True
    else:
        return False

class TikTakToe(Node):
    def __init__(self):
        super().__init__("dummy_server_py")

        self.server_ = self.create_service(
            Index, "dummy_service", self.callback_service_call
        )
        self.get_logger().info("Dummy service server node started.")

    def callback_service_call(self, request, response):
        result = self.algorithm(request.index)
        print(result)
        if len(empty_cells(board)) == 0 or game_over(board):
            if wins(board, HUMAN):
                print('YOU WIN!')
            elif wins(board, COMP):
                print('YOU LOSE!')
            else:
                print('DRAW!')
            exit()
        response = True
        return response
    
    def algorithm(self, index):
        o=0
        x=0
        for i in range(9):
            if index[i] == 1:              # 1 is O
                o += 1
            elif index[i] == 2:            # 2 is X
                x += 1
            else:                          # 0 is empty
                pass
        for i in range(9):
            if o>x:
                if index[i] == 1:
                    set_move(i//3, i%3, HUMAN)
                elif index[i] == 2:
                    set_move(i//3, i%3, COMP)
                else:
                    pass
            else:
                if index[i] == 1:
                    set_move(i//3, i%3, COMP)
                elif index[i] == 2:
                    set_move(i//3, i%3, HUMAN)
                else:
                    pass
        return ai_turn()

def main(args=None):
    rclpy.init(args=args) 
    node = TikTakToe()  
    rclpy.spin(node)  
    rclpy.shutdown()  


if __name__ == "__main__":
    main()
