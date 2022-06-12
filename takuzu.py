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

# "Util" function which isn't in utils.py (and we can't import math)
def ceiling_division(dividend: int, divisor: int) -> int:
    """Returns the ceiling of dividend / divisor."""
    return (dividend + divisor - 1) // divisor


class TakuzuState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = TakuzuState.state_id
        TakuzuState.state_id += 1

    def __eq__(self, other) -> bool:
        return self.id == other.id

    def __lt__(self, other):
        return self.id < other.id


class Board:
    """Representação interna de um tabuleiro de Takuzu."""

    EMPTY_CELL = 2

    def __init__(self, board: list, size: int) -> None:
        self.size = size
        self.board = board
        actions = self.empty_cells()
        self.possible_actions = [
            (row, col, value) for row, col in actions for value in (0, 1)
        ]

    def get_number(self, row: int, col: int) -> int:
        """Devolve o valor na respetiva posição do tabuleiro."""
        return self.board[row][col]

    def get_column_count(self, col: int) -> (int, int):
        """Devolve o número de 0's e 1's na coluna especificada."""
        count = [0, 0]
        for row in range(self.size):
            val = self.get_number(row, col)
            if val == self.EMPTY_CELL:
                continue
            count[val] += 1
        return count

    def get_row_count(self, row: int) -> (int, int):
        """Devolve o número de 0's e 1's na linha especificada."""
        count = [0, 0]
        for col in range(self.size):
            val = self.get_number(row, col)
            if val == self.EMPTY_CELL:
                continue
            count[val] += 1
        return count

    def fill_cell(self, row: int, col: int, value: int) -> None:
        """Preenche uma célula com um valor."""
        list_board = list(self.board)
        list_board[row] = list(self.board[row])
        list_board[row][col] = value
        list_board[row] = tuple(list_board[row])
        self.board = tuple(list_board)
        self.possible_actions.remove((row, col, value))
        # FIXME: can't remove it, since something is wrong with the DFS
        # try:
        self.possible_actions.remove((row, col, 1 - value))
        # except ValueError:
        # pass

    def empty_cells(self) -> list:
        """Devolve uma lista com as posições vazias do tabuleiro."""
        return [
            (row, col)
            for row in range(self.size)
            for col in range(self.size)
            if self.board[row][col] == self.EMPTY_CELL
        ]

    def adjacent_vertical_numbers(self, row: int, col: int) -> (int, int):
        """Devolve os valores imediatamente abaixo e acima,
        respectivamente."""
        adj = [None, None]
        if row - 1 >= 0:
            adj[0] = self.board[row - 1][col]
        if row + 1 < self.size:
            adj[1] = self.board[row + 1][col]
        return tuple(adj)

    def adjacent_horizontal_numbers(self, row: int, col: int) -> (int, int):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        adj = [None, None]
        if col - 1 >= 0:
            adj[0] = self.board[row][col - 1]
        if col + 1 < self.size:
            adj[1] = self.board[row][col + 1]
        return tuple(adj)

    def check_3_straight(self, row: int, col: int, val: int) -> bool:
        """Checks whether the action creates a 3 in a row situation."""
        to_avoid = (val, val)

        def checker(line: int, possibilities: list) -> bool:
            # Considering a line such as 1-0-2-1-0
            if self.size < 3:
                return False
            elif line == 0:
                # would check (0, 2)
                return possibilities[2] == to_avoid
            elif line == self.size - 1:
                # would check (2, 1)
                return possibilities[0] == to_avoid
            return any(possibilities[i] == to_avoid for i in range(0, 3))

        vertical_adjacencies = [
            self.adjacent_vertical_numbers(row + i, col) for i in range(-1, 2)
        ]
        horizontal_adjacencies = [
            self.adjacent_horizontal_numbers(row, col + i) for i in range(-1, 2)
        ]
        return checker(row, vertical_adjacencies) or checker(
            col, horizontal_adjacencies
        )

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
        board = []
        for line in sys.stdin.readlines():
            board.append(tuple(map(int, line.split())))
        return Board(board, n)

    # copy of the board
    def __copy__(self):
        return Board(self.board, self.size)

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
        possible_actions = []
        for action in state.board.possible_actions:
            if not self.impossible(action, state.board):
                possible_actions.append(action)
        return possible_actions

    def result(self, state: TakuzuState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        row, col, value = action
        new_board = state.board.__copy__()
        new_board.fill_cell(row, col, value)
        print("Board is now:\n{}".format(new_board))
        return TakuzuState(new_board)

    def goal_test(self, state: TakuzuState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas com uma sequência de números adjacentes."""
        # we reached a goal_state if there are no possible actions left - there are no cells with value 2 (missing stuff)
        return len(state.board.possible_actions) == 0 and not any(
            state.board.board[row][col] == 2
            for row in range(state.board.size)
            for col in range(state.board.size)
        )

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
        row_count, col_count = board.get_row_count(row), board.get_column_count(col)
        ceiling = ceiling_division(board.size, 2)
        if row_count[value] >= ceiling or col_count[value] >= ceiling:
            return True
        return board.check_3_straight(row, col, value)
        # TODO: return board.check_2_equal_rows_or_columns(row, col, value)

    def mandatory(self, action: tuple, board: Board) -> bool:
        """Checks whether the action is mandatory or not (it placing a value in
        those coordinates will always have to happen, given the current
        board configuration."""
        row, col, value = action
        return self.impossible((row, col, 1 - value), board) and not self.impossible(
            (row, col, value), board
        )


if __name__ == "__main__":
    board = Board.parse_instance_from_stdin()
    takuzu = Takuzu(board)
    goal = depth_first_tree_search(takuzu)
    print(goal.state.board)
