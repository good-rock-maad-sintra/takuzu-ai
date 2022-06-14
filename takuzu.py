# takuzu.py: Template para implementação do projeto de Inteligência Artificial 2021/2022.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 5:
# 99207 Diogo Gaspar
# 99256 João Rocha

import sys
import numpy as np
from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_graph_search,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
    compare_searchers
)

def debug(state):
    print("Doing action: {}".format(state.action))
    print("Board is currently:\n{}".format(state.board))

# "Util" function which isn't in utils.py (and we can't import math)
def ceiling_division(dividend: int, divisor: int) -> int:
    """Returns the ceiling of dividend / divisor."""
    return (dividend + divisor - 1) // divisor


def tuple_assignment(row: int, col: int, value: int, t: tuple):
    return t[0:row] + (t[row][0:col] + (value,) + t[row][col + 1 :],) + t[row + 1 :]


class TakuzuState:
    state_id = 0

    def __init__(self, board, action=None):
        self.board = board
        self.action = action
        self.id = TakuzuState.state_id
        TakuzuState.state_id += 1
    
    def __eq__(self, other) -> bool:
        return self.id == other.id

    def __lt__(self, other) -> bool:
        return self.id < other.id

    def __hash__(self):
        return hash(self.id)


class Board:
    """Representação interna de um tabuleiro de Takuzu."""

    EMPTY_CELL = 2

    def __init__(self, board: list, size: int, initial=False) -> None:
        self.size = size
        self.board = board
        if initial:
            self.empty_cells = self.empty_cells()
            self.columns = set()
            self.rows = set()
            for x in range(self.size):
                if self.full_check(self.get_row_count(x)):
                    self.rows.add(self.get_bin_row(x))
                if self.full_check(self.get_col_count(x)):
                    self.columns.add(self.get_bin_col(x))

    def new_board(self, x: int, y: int, val: int):
        """Preenche uma célula com um valor."""
        aux = [[col for col in row] for row in self.board]
        aux[x][y] = val
        new_board = Board(aux, self.size)

        new_board.empty_cells = self.empty_cells.copy()
        new_board.rows = self.rows.copy()
        new_board.columns = self.columns.copy()
        
        new_board.empty_cells.remove((x, y))
        if new_board.full_check(new_board.get_row_count(x)):
            new_board.rows.add(new_board.get_bin_row(x))
        if new_board.full_check(new_board.get_col_count(y)):
            new_board.columns.add(new_board.get_bin_col(y))

        return new_board

    def get_number(self, row: int, col: int) -> int:
        """Devolve o valor na respetiva posição do tabuleiro."""
        if not 0 <= row < self.size or not 0 <= col < self.size:
            return None
        return self.board[row][col]

    def get_col_count(self, col: int):
        """Devolve o número de 0's e 1's na coluna especificada."""
        count = [0, 0]
        for row in range(self.size):
            val = self.get_number(row, col)
            if val == self.EMPTY_CELL:
                continue
            count[val] += 1
        return count

    def get_row_count(self, row: int):
        """Devolve o número de 0's e 1's na linha especificada."""
        count = [0, 0]
        for col in range(self.size):
            val = self.get_number(row, col)
            if val == self.EMPTY_CELL:
                continue
            count[val] += 1
        return count

    def full_check(self, count: tuple):
        return count[0] + count[1] == self.size

    def almost_full_check(self, count: tuple):
        return count[0] + count[1] == self.size - 1

    def get_bin_row(self, row: int, action=None):
        res = 0b0
        for x in range(self.size):
            if self.get_number(row, x) == self.EMPTY_CELL:
                if action != None and action[0] == row and action[1] == x:
                    res |= (action[2] << x)
                else:
                    raise ValueError
            else:
                res |= (self.get_number(row, x) << x)
        return res

    def get_bin_col(self, col: int, action=None):
        res = 0b0
        for x in range(self.size):
            if self.get_number(x, col) == self.EMPTY_CELL:
                if action != None and action[0] == x and action[1] == col:
                    res |= (action[2] << x)
                else:
                    raise ValueError
            else:
                res |= (self.get_number(x, col) << x)
        return res

    def empty_cells(self) -> list:
        """Devolve uma lista com as posições vazias do tabuleiro."""
        return [
            (row, col)
            for row in range(self.size)
            for col in range(self.size)
            if self.board[row][col] == self.EMPTY_CELL
        ]

    def adjacent_vertical_numbers(self, row: int, col: int):
        """Devolve os valores imediatamente abaixo e acima,
        respectivamente."""
        if not 0 <= row < self.size or not 0 <= col < self.size:
            return None
        return (self.get_number(row - 1, col), self.get_number(row + 1, col))

    def adjacent_horizontal_numbers(self, row: int, col: int):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        if not 0 <= row < self.size or not 0 <= col < self.size:
            return None
        return (self.get_number(row, col - 1), self.get_number(row, col + 1))

    def check_3_straight(self, row: int, col: int, val: int) -> bool:
        """Checks whether the action creates a 3 in a row situation."""
        to_avoid = (val, val)
        vertical_adjacencies = [
            (self.get_number(row - 2, col), self.get_number(row - 1, col)),
            self.adjacent_vertical_numbers(row, col),
            (self.get_number(row + 1, col), self.get_number(row + 2, col)),
        ]
        horizontal_adjacencies = [
            (self.get_number(row, col - 2), self.get_number(row, col - 1)),
            self.adjacent_horizontal_numbers(row, col),
            (self.get_number(row, col + 1), self.get_number(row, col + 2)),
        ]
        return to_avoid in vertical_adjacencies + horizontal_adjacencies

    @staticmethod
    def parse_instance_from_stdin():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.

        Por exemplo:
            $ python3 takuzu.py < input_T01

            > from sys import stdin
            > stdin.readline()
        """
        n = int(sys.stdin.readline())
        board = ()
        for line in sys.stdin.readlines():
            board += (list(map(int, line.split())),)
        return Board(board, n, initial=True)

    def __str__(self):
        """Imprime o tabuleiro."""
        return "\n".join(["\t".join(map(str, row)) for row in self.board])


class Takuzu(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        self.initial = TakuzuState(board)

    def actions(self, state: TakuzuState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        ran_once = False
        possible = []
        for row, col in state.board.empty_cells:
            ac0, ac1 = (row, col, 0), (row, col, 1)
            if self.mandatory(ac0, state):
                return [ac0]
            elif self.mandatory(ac1, state):
                return [ac1]

            if not ran_once:
                if self.possible(ac0, state):
                    possible.append(ac0)
                if self.possible(ac1, state):
                    possible.append(ac1)
                ran_once = True
        return possible

    def result(self, state: TakuzuState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        return TakuzuState(state.board, action)

    def goal_test(self, state: TakuzuState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas com uma sequência de números adjacentes."""
        if state.action:
            x, y, val = state.action
            state.board = state.board.new_board(x, y, val)
        
        # debug(state)
        return len(state.board.empty_cells) == 0

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""

        def line_heuristic(counts: list) -> float:
            return 1 / (counts[0] + counts[1])

        action = node.action
        if action == None:
            return 0
        board = node.state.board
        if self.impossible(action, node.state):
            return 1
        elif self.mandatory(action, node.state):
            return 0

        result = 0
        for x,y in board.empty_cells:
            row_count, col_count = board.get_row_count(x), board.get_col_count(y)
            result += line_heuristic(row_count) + line_heuristic(col_count)
        return result / (2*board.size)

    def action_equals_row(self, row: int, row_count: list, col: int, col_count: list, action: tuple, board: list):
        """Checks whether the action creates a situation where there are two
        equal rows and/or columns in the board (fully filled)."""
        return (
            (board.almost_full_check(row_count) and board.get_bin_row(row, action) in board.rows)
            or
            (board.almost_full_check(col_count) and board.get_bin_col(col, action) in board.columns)
        )
    
    def impossible(self, action: tuple, state: TakuzuState) -> bool:
        """Checks whether executing the action is impossible or not."""
        board = state.board
        row, col, value = action
        if board.get_number(row, col) != board.EMPTY_CELL:
            return True
        if board.check_3_straight(row, col, value):
            return True

        row_count, col_count = board.get_row_count(row), board.get_col_count(col)
        ceiling = ceiling_division(board.size, 2)
        if row_count[value] >= ceiling or col_count[value] >= ceiling:
            return True
        if self.action_equals_row(row, row_count, col, col_count, action, board):
            return True
        return False

    def possible(self, action: tuple, state: TakuzuState) -> bool:
        """Checks whether executing the action is possible or not."""
        return not self.impossible(action, state)

    def mandatory(self, action: tuple, state: TakuzuState) -> bool:
        """Checks whether the action is mandatory or not (it placing a value in
        those coordinates will always have to happen, given the current
        state configuration."""
        row, col, value = action
        return self.impossible((row, col, 1 - value), state) and \
                self.possible((row, col, value), state)


if __name__ == "__main__":
    board = Board.parse_instance_from_stdin()
    takuzu = Takuzu(board)
    goal = depth_first_tree_search(takuzu)
    #print("---")
    if goal:
        print(goal.state.board)
    else:
        print('No goal')
