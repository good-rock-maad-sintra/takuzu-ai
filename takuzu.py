# takuzu.py: Template para implementação do projeto de Inteligência Artificial 2021/2022.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 5:
# 99207 Diogo Gaspar
# 99256 João Rocha

import sys
from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
)

# "Util" function which isn't in utils.py (and we can't import math)
# Useful because of cases where the board has an odd size
def ceiling(n: int, m: int) -> int:
    """Returns the ceiling of n // m."""
    return (n + m - 1) // m

class TakuzuState:
    state_id = 0

    def __init__(self, board) -> None:
        self.board = board
        self.id = TakuzuState.state_id
        TakuzuState.state_id += 1
    
    def __eq__(self, other) -> bool:
        return self.id == other.id

    def __lt__(self, other) -> bool:
        return self.id < other.id

    def __hash__(self) -> int:
        return hash(self.id)


class Board:
    """Internal representation of a Takuzu board."""
    EMPTY_CELL = 2

    def __init__(self, board, size, action=None) -> None:
        """The constructor can be called in one of two ways:
        - with a board (a list) and a size (an int), to construct a Board object for
        the first time;
        - with a board (an object of the class Board), a size (an int) and an action,
        to construct a Board object from the previous one, after performing 'action'"""
        self.size = size
        if not action:
            self.board = board
            self.empty_cells = [
                (row, col)
                for row in range(self.size)
                for col in range(self.size)
                if self.board[row][col] == self.EMPTY_CELL
            ]
            self.columns = set()
            self.rows = set()
            for x in range(self.size):
                if self.full_check(self.get_row_count(x)):
                    self.rows.add(self.get_bin_row(x))
                if self.full_check(self.get_col_count(x)):
                    self.columns.add(self.get_bin_col(x))
            return

        x, y, value = action
        self.board = [[cell for cell in row] for row in board.board]
        self.board[x][y] = value

        self.empty_cells = board.empty_cells.copy()
        self.rows = board.rows.copy()
        self.columns = board.columns.copy()

        self.empty_cells.remove((x, y))
        if self.full_check(self.get_row_count(x)):
            self.rows.add(self.get_bin_row(x))
        if self.full_check(self.get_col_count(y)):
            self.columns.add(self.get_bin_col(y))

    def get_number(self, row: int, col: int) -> int:
        """Returns the value in the board with positions (row, col)."""
        if not 0 <= row < self.size or not 0 <= col < self.size:
            return None
        return self.board[row][col]

    def adjacent_vertical_numbers(self, row: int, col: int) -> tuple:
        """Returns the values of the cells imediately above and below of (row, col)."""
        if not 0 <= row < self.size or not 0 <= col < self.size:
            return None
        return (self.get_number(row - 1, col), self.get_number(row + 1, col))

    def adjacent_horizontal_numbers(self, row: int, col: int) -> tuple:
        """Returns the values of the cells imediately right and left of (row, col)."""
        if not 0 <= row < self.size or not 0 <= col < self.size:
            return None
        return (self.get_number(row, col - 1), self.get_number(row, col + 1))

    def check_3_straight(self, row: int, col: int, val: int) -> bool:
        """Checks whether the action given by (row, col, val) creates a 3 in
        a row situation."""
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

    def get_row_count(self, row: int) -> list:
        """Returns the amount of 0's and 1's in the specified row."""
        count = [0, 0]
        for col in range(self.size):
            val = self.get_number(row, col)
            if val == self.EMPTY_CELL:
                continue
            count[val] += 1
        return count

    def get_col_count(self, col: int) -> list:
        """Returns the amount of 0's and 1's in the specified column."""
        count = [0, 0]
        for row in range(self.size):
            val = self.get_number(row, col)
            if val == self.EMPTY_CELL:
                continue
            count[val] += 1
        return count

    def get_bin_row(self, row: int, action=None) -> int:
        """Returns a binary representation of a row. If action=None, returns
        the representation of the row 'row', otherwise returns its representation
        after performing the given action. This function can only be called on
        rows which are either full or will be full after performing action."""
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

    def get_bin_col(self, col: int, action=None) -> int:
        """Returns a binary representation of a column. If action=None returns
        the representation of the column 'col', otherwise returns its representation
        after performing action. This function can only be called on columns
        which are either full or will be full after performing action."""
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

    def full_check(self, count: tuple) -> bool:
        """Checks if a line is full (that is, with no empty cells)."""
        return count[0] + count[1] == self.size

    def almost_full_check(self, count: tuple) -> bool:
        """Checks if a line is almost full (that is, with one empty cell)."""
        return count[0] + count[1] == self.size - 1

    def action_creates_equal_lines(self, row_count: list, col_count: list, action: tuple) -> bool:
        """Checks whether 'action' creates a situation where there are two equal
        rows and/or columns in the board (fully filled)."""
        row, col, _ = action
        return (
            (self.almost_full_check(row_count) and \
            self.get_bin_row(row, action) in self.rows) or \
            (self.almost_full_check(col_count) and \
            self.get_bin_col(col, action) in self.columns)
        )

    @staticmethod
    def parse_instance_from_stdin():
        """Reads the test from the standard input (stdin), passed as an argument,
        and returns a Board instance.

        For example:
            $ python3 takuzu.py < input_T01

            > from sys import stdin
            > stdin.readline()
        """
        n = int(sys.stdin.readline())
        board = ()
        for line in sys.stdin.readlines():
            board += (list(map(int, line.split())),)
        return Board(board, n)

    def __str__(self) -> str:
        """Prints the board."""
        return "\n".join(["\t".join(map(str, row)) for row in self.board])


class Takuzu(Problem):
    def __init__(self, board: Board) -> None:
        """The constructor specifies the initial state."""
        self.initial = TakuzuState(board)

    def actions(self, state: TakuzuState) -> list:
        """Returns a list of actions which can be executed from 'state'."""
        ran_once = False
        possible = []
        for row, col in state.board.empty_cells:
            ac0, ac1 = (row, col, 0), (row, col, 1)
            if self.impossible(ac0, state) and self.impossible(ac1, state):
                return []
            elif self.mandatory(ac0, state):
                return [ac0]
            elif self.mandatory(ac1, state):
                return [ac1]

            if not ran_once:
                possible = [ac0, ac1]
                ran_once = True
        return possible

    def result(self, state: TakuzuState, action) -> TakuzuState:
        """Returns the resulting state of executing action over 'state'.
        'action' should be present in the list returned by self.actions(state)."""
        board = state.board
        new_board = Board(board, board.size, action)
        return TakuzuState(new_board)

    def goal_test(self, state: TakuzuState) -> bool:
        """Returns True if (and only if) 'state' is a goal state. Should check
        whether all the board's cells are filled with a sequence of adjacent
        numbers."""
        return len(state.board.empty_cells) == 0

    def h(self, node: Node) -> float:
        """Heuristic function utilized for the A* search."""

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

    def impossible(self, action: tuple, state: TakuzuState) -> bool:
        """Checks whether executing 'action' is impossible or not - that is,
        if the cell where it would be executed is already filled, if it
        creates a 3-in-a-row situation, if the row and/or column where the
        action would be executed already has the maximum amount of the value
        possible, or if it creates a situation where two fully filled rows
        or columns are equal."""
        board = state.board
        row, col, value = action
        if board.get_number(row, col) != board.EMPTY_CELL:
            return True
        if board.check_3_straight(row, col, value):
            return True

        row_count, col_count = board.get_row_count(row), board.get_col_count(col)
        ceil = ceiling(board.size, 2)
        if row_count[value] >= ceil or col_count[value] >= ceil:
            return True
        if board.action_creates_equal_lines(row_count, col_count, action):
            return True
        return False

    def possible(self, action: tuple, state: TakuzuState) -> bool:
        """Checks whether executing 'action' is possible or not - that is, if
        it's not impossible."""
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
    if goal:
        print(goal.state.board)
    else:
        print('No goal')
