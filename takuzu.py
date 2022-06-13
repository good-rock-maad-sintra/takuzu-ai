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
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
)

# TODO: remove this
from time import sleep


# "Util" function which isn't in utils.py (and we can't import math)
def ceiling_division(dividend: int, divisor: int) -> int:
    """Returns the ceiling of dividend / divisor."""
    return (dividend + divisor - 1) // divisor


def tuple_assignment(row: int, col: int, value: int, t: tuple):
    return t[0:row] + (t[row][0:col] + (value,) + t[row][col + 1 :],) + t[row + 1 :]


class TakuzuState:
    state_id = 0
    action = None

    def __init__(self, board, parent_actions=None, action=None):
        self.board = board
        self.action = action
        if self.action is None:
            self.actions = [
                (row, col, value) for row, col in self.board.empty_cells()
                        for value in (0, 1)
            ]
        else:
            self.actions = parent_actions
        self.id = TakuzuState.state_id
        TakuzuState.state_id += 1

    def __eq__(self, other) -> bool:
        return self.id == other.id

    def __lt__(self, other) -> bool:
        return self.id < other.id


class Board:
    """Representação interna de um tabuleiro de Takuzu."""

    EMPTY_CELL = 2

    def __init__(self, board: tuple, size: int) -> None:
        self.size = size
        self.board = board

    def get_number(self, row: int, col: int) -> int:
        """Devolve o valor na respetiva posição do tabuleiro."""
        if not 0 <= row < self.size or not 0 <= col < self.size:
            return None
        return self.board[row][col]

    def get_column_count(self, col: int):
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

    def fill_cell(self, row: int, col: int, value: int):
        """Preenche uma célula com um valor."""
        aux = tuple_assignment(row, col, value, self.board)
        return Board(aux, self.size)

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
            board += (tuple(map(int, line.split())),)
        return Board(board, n)

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
        actions = [
            action for action in state.actions \
                    if self.possible(action, state.board)
        ]
        state.actions = actions
        return actions
        """
        mandatory_actions = []
        possible_actions = []
        for action in state.possible_actions:
            if self.mandatory(action, state.board):
                mandatory_actions.append(action)
            elif self.possible(action, state.board):
                possible_actions.append(action)
        return mandatory_actions if mandatory_actions != [] else possible_actions
        """

    def result(self, state: TakuzuState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        row, col, value = action
        new_board = state.board.fill_cell(row, col, value)
        return TakuzuState(new_board, state.actions, action)

    def goal_test(self, state: TakuzuState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas com uma sequência de números adjacentes."""
        # we reached a goal_state if there are no possible actions left - there are no cells with value 2 (missing stuff)
        print("Doing action: {}".format(state.action))
        print("Board is currently:\n{}".format(state.board))
        print("Possible actions: {}".format(self.actions(state)))
        return len(state.board.empty_cells()) == 0

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""

        def calc_heuristic(size: int, counts: list) -> float:
            zero_count, one_count = counts
            return (size / 2 - zero_count) / (size - zero_count - one_count)

        action = node.action
        board = node.state.board
        if self.impossible(action, board):
            return 0
        elif self.mandatory(action, board):
            return 1
        row, col, _ = action
        row_count, col_count = board.get_row_count(row), board.get_col_count(col)
        return calc_heuristic(board.size, row_count) + calc_heuristic(
            board.size, col_count
        )

    def impossible(self, action: tuple, board: Board) -> bool:
        """Checks whether executing the action is impossible or not."""
        row, col, value = action
        if board.get_number(row, col) != board.EMPTY_CELL:
            return True
        if board.check_3_straight(row, col, value):
            return True
        row_count, col_count = board.get_row_count(row), board.get_column_count(col)
        ceiling = ceiling_division(board.size, 2)
        if row_count[value] >= ceiling or col_count[value] >= ceiling:
            return True
        return False
        # TODO: return board.check_2_equal_rows_or_columns(row, col, value)

    def possible(self, action: tuple, board: Board) -> bool:
        """Checks whether executing the action is possible or not."""
        return not self.impossible(action, board)

    def mandatory(self, action: tuple, board: Board) -> bool:
        """Checks whether the action is mandatory or not (it placing a value in
        those coordinates will always have to happen, given the current
        board configuration."""
        row, col, value = action
        return self.impossible((row, col, 1 - value), board) and self.possible(
            (row, col, value), board
        )


if __name__ == "__main__":
    board = Board.parse_instance_from_stdin()
    takuzu = Takuzu(board)
    print(board)
    print(board.empty_cells())
    goal = depth_first_tree_search(takuzu)
    print("---")
    print(goal.state.board)
