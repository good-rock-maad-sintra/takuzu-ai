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

    def __init__(self, board, parent_mandatory_actions=None, \
            parent_possible_actions=None, action=None, \
            parent_rows=None, parent_columns=None):
        self.board = board
        self.action = action

        if self.action is None:
            self.mandatory_actions = set()
            self.possible_actions = set()
            self.columns = set()
            self.rows = set()
            for x in range(self.board.size):
                if self.board.full_check(self.board.get_row_count(x)):
                    self.rows.add(self.board.get_bin_row(x))
                if self.board.full_check(self.board.get_col_count(x)):
                    self.columns.add(self.board.get_bin_col(x))
        else:
            self.mandatory_actions = parent_mandatory_actions.copy()
            self.possible_actions = parent_possible_actions.copy()
            if self.action in self.mandatory_actions:
                self.mandatory_actions.remove(self.action)
            elif self.action in self.possible_actions:
                self.possible_actions.remove(self.action)

            self.rows = parent_rows.copy()
            self.columns = parent_columns.copy()
            row, col, _ = self.action
            if self.board.almost_full_check(self.board.get_row_count(row)):
                self.rows.add(self.board.get_bin_row(row, self.action))
            if self.board.almost_full_check(self.board.get_col_count(col)):
                self.columns.add(self.board.get_bin_col(col, self.action))

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

    def __init__(self, board: tuple, size: int) -> None:
        self.size = size
        self.board = board

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
                if action == None:
                    raise ValueError
                elif action[0] == row and action[1] == x:
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
                if action == None:
                    raise ValueError
                elif action[0] == x and action[1] == col:
                    res |= (action[2] << x)
                else:
                    raise ValueError
            else:
                res |= (self.get_number(x, col) << x)
        return res

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
        if state.action == None:
            for x,y in state.board.empty_cells():
                for val in range(2):
                    action = (x,y,val)
                    if self.mandatory(action, state):
                        state.mandatory_actions.add(action)
                    elif self.possible(action, state):
                        state.possible_actions.add(action)
        else:
            new_mand_actions = set()
            for action in state.mandatory_actions:
                if self.possible(action, state):
                    return [action]
                else:
                    return []

            new_poss_actions = set()
            for action in state.possible_actions:
                row, col, value = action
                conj_action = (row, col, 1 - value)
                if conj_action == state.action:
                    continue

                if self.mandatory(action, state):
                    new_mand_actions.add(action)
                elif self.possible(action, state):
                    new_poss_actions.add(action)

            state.mandatory_actions = new_mand_actions
            state.possible_actions = new_poss_actions

        if len(state.mandatory_actions) > 0:
            return [state.mandatory_actions.pop()]
        return list(state.possible_actions)

    def result(self, state: TakuzuState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        row, col, value = action
        return TakuzuState(state.board, state.mandatory_actions, state.possible_actions, action, state.rows, state.columns)

    def goal_test(self, state: TakuzuState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas com uma sequência de números adjacentes."""
        if state.action:
            x, y, val = state.action
            state.board = state.board.fill_cell(x, y, val)
        
        #debug(state)
        return len(state.board.empty_cells()) == 0

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""

        def line_heuristic(counts: list) -> float:
            return 1 / (count[0] + count[1])

        action = node.action
        if action == None:
            return 0
        board = node.state.board
        if self.impossible(action, node.state):
            return 1
        elif self.mandatory(action, node.state):
            return 0

        result = 0
        for x,y in board.empty_cells():
            row_count, col_count = board.get_row_count(x), board.get_col_count(y)
            result += line_heuristic(row_count) + line_heuristic(col_count)
        return result / (2*n)

    def impossible(self, action: tuple, state: TakuzuState) -> bool:
        """Checks whether executing the action is impossible or not."""
        board = state.board
        row, col, value = action
        if board.get_number(row, col) != board.EMPTY_CELL:
            return True
        if board.check_3_straight(row, col, value):
            return True

        row_count, col_count = board.get_row_count(row), board.get_col_count(col)
        if (board.almost_full_check(row_count) and \
                board.get_bin_row(row, action) in state.rows) or \
                (board.almost_full_check(col_count) and \
                board.get_bin_col(col, action) in state.columns):
            return True
        ceiling = ceiling_division(board.size, 2)
        if row_count[value] >= ceiling or col_count[value] >= ceiling:
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
    goal = breadth_first_tree_search(takuzu)
    #print("---")
    if goal:
        print(goal.state.board)
    else:
        print('No goal')
