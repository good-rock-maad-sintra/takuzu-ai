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


class TakuzuState:
    state_id = 0
    column_n = 0
    row_n = 0
    empty_cells = 0

    def __init__(self, board):
        self.board = board
        self.columns = set()
        self.rows = set()
        self.id = TakuzuState.state_id
        self.empty_cells = self.board.get_empty_cells_n()
        TakuzuState.state_id += 1

    def __eq__(self, other):
        return self.id == other.id

    def __lt__(self, other):
        return self.id < other.id

class Board:
    """Representação interna de um tabuleiro de Takuzu."""
    size = 0
    board = []
    EMPTY_CELL = 2

    def __init__(self, size, board) -> None:
        self.size = size
        self.board = board

    def get_number(self, row: int, col: int) -> int:
        """Devolve o valor na respetiva posição do tabuleiro."""
        if not 0 <= row < self.size or not 0 <= col < self.size:
            return None
        return self.board[row][col]

    def get_row(self, row: int):
        return self.board[row]

    def get_column(self, col: int):
        return [self.board[x][col] for x in range(self.size)]

    def get_empty_cells_n(self):
        count = 0
        for x in range(self.size):
            for y in range(self.size):
                if self.cell_empty(x,y):
                    count+=1
        return count

    def adjacent_vertical_numbers(self, row: int, col: int) -> (int, int):
        """Devolve os valores imediatamente abaixo e acima,
        respectivamente."""
        if not 0 <= row < self.size or not 0 <= row < self.size:
            return None
        return (self.get_number(row-1, col), self.get_number(row+1, col))

    def adjacent_horizontal_numbers(self, row: int, col: int) -> (int, int):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        if not 0 <= col < self.size or not 0 <= col < self.size:
            return None
        return (self.get_number(row, col-1), self.get_number(row, col+1))

    def count_row(self, row: int) -> (int, int):
        counts = [0, 0]
        for y in range(self.size):
            val = self.get_number(row, y)
            if val in (0, 1):
                counts[val] += 1
        return tuple(counts)

    def count_column(self, col: int) -> (int, int):
        counts = [0, 0]
        for x in range(self.size):
            val = self.get_number(x, col)
            if val in (0, 1):
                counts[val] += 1
        return tuple(counts)
    
    def check_3_straight(self, row: int, col: int, val: int) -> bool:
        """Verifica se ao colocar um valor numa posição cria uma situação
        de 3 valores iguais adjacentes - True indica que cria."""
        to_avoid = (val, val)
        def checker(line: int, possibilities: list):
            if self.size == 1:
                return False
            elif line == 0:
                return possibilities[2] == to_avoid
            elif line == self.size - 1:
                return possibilities[0] == to_avoid
            return any(possibilities[i] == to_avoid for i in range(3))
        
        vertical_adjacencies = [self.adjacent_vertical_numbers(row + i, col) for i in range(-1, 2)]
        horizontal_adjacencies = [self.adjacent_horizontal_numbers(row, col + i) for i in range(-1, 2)]
        return checker(row, vertical_adjacencies) or checker(col, horizontal_adjacencies)

    def cell_empty(self, row, col):
        return self.get_number(row, col) == self.EMPTY_CELL

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
            board.append(list(map(int, line.split())))
        return Board(n, board)
    
    def transpose_board(self):
        return np.transpose(self.board)
    
    def fill_cell(self, row: int, col: int, val: int):
        """Coloca um valor numa posição do tabuleiro."""
        self.board[row][col] = val

    def __str__(self) -> str:
        return "\n".join(["\t".join(map(str, row)) for row in self.board])


class Takuzu(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        self.initial = TakuzuState(board)
        self.moves = self.actions(self.initial)

    def actions(self, state: TakuzuState):
        board = state.board
        moves = []
        for x in range(board.size):
            for y in range(board.size):
                if board.cell_empty(x,y):
                    for k in range(2):
                        if self.possible(state, (x,y,k)):
                            moves.append((x,y,k))
        print(moves)
        return moves

    def result(self, state: TakuzuState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        x, y, val = action
        updated_board = Board(state.board.size, state.board.board)
        updated_board.fill_cell(x, y, val)
        return TakuzuState(updated_board)

    def consistent_test(self, state: TakuzuState):
        #print(len(state.columns) == state.column_n and \
        #        len(state.rows) == state.row_n)
        return len(state.columns) == state.column_n and \
                len(state.rows) == state.row_n

    def goal_test(self, state: TakuzuState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas com uma sequência de números adjacentes."""
        print(state.empty_cells)
        return state.empty_cells == 0 and self.consistent_test(state)
            
    def impossible(self, state: TakuzuState, action):
        """Verifica se executar a ação-argumento leva a um estado em que é
        impossível completar o tabuleiro segundo as regras do jogo."""
        hyp_state = self.result(state, action)
        row, col, val = action
        cap = (state.board.size+1) // 2 # produces ceiling
        ct_col = board.count_column(col)
        ct_row = board.count_row(row)
        return ct_col[val] > cap or ct_row[val] > cap or \
                hyp_state.board.check_3_straight(row, col, val)

    def possible(self, state: TakuzuState, action):
        return not self.impossible(state, action)

    def mandatory(self, state: TakuzuState, action):
        """Verifica se executar a ação-argumento é obrigatório - ou seja,
        se não é possível colocar outro valor na posição pretendida."""
        conj_action = (action[0], action[1], 1-action[2])
        return self.impossible(state, conj_action) and \
                self.possible(state, action)

    def h(self, node: Node):
        def adjacency_tendency_compute(adjacent_pair):
            x, y = adjacent_pair
            if (x,y) == (2,2) or (x,y) == (0,1) or (x,y) == (1,0):
                return 1/2
            elif (x,y) == (1,2) or (x,y) == (2,1):
                return 1
            elif (x,y) == (0,2) or (x,y) == (2,0):
                return 0

        def adjacency_tendency(node: Node, action):
            board = node.state.board
            x, y, val = action
            hor_adjacent = board.adjacent_horizontal_numbers(x, y)
            ver_adjacent = board.adjacent_vertical_numbers(x, y)
            return (adjacent_tendency_compute(node, hor_adjacent) + \
                    adjacent_tendency_compute(node, hor_adjacent)) / 2

        #def line_tendency(node: Node, action):
            # TODO

        moves = self.actions(node.state)
        if self.mandatory(node, moves[0]):
            return 0
        elif self.impossible(node, moves[0]):
            return 1
        return 1
    
if __name__ == "__main__":
    board = Board.parse_instance_from_stdin()
    # print(board)
    takuzu = Takuzu(board)
    # obviamente depois a estratégia varia, e temos de testar várias
    goal = depth_first_tree_search(takuzu)
    if goal == None:
        print('desagradavel')
    else:
        print(goal.state.board)

"""
    def backtrack(self):
        (state, action) = self.decision_states.pop()
        x, y, val = action
        conj_action = (x, y, 1-val)
        if possible(state, conj_action):
            return result(state, conj_action)
        return 

    def decide(self, state: TakuzuState):
        self.moves = self.actions(state)

        if len(self.moves) == 0:
            return None
        else:
            action = self.moves.pop()
            # Action is not mandatory, thus we made a choice
            if h(action) != 0:
                self.decision_states.add((curr_state, action))
            return action

    def solve(self):
        curr_state = self.initial

        while not self.goal_test(curr_state):
            # Check CSP consistency of current state
            if not self.consistent_test(curr_state):
                if len(self.decision_states) == 0:
                # First decision state is inconsistent, thus problem isn't solvable
                    return FAILURE
                curr_state = self.backtrack(dec_state, action)
                continue

            # Decide which action to take
            action = self.decide(curr_state)
            if action != None:
                curr_state = self.result(curr_state, action)

        return curr_state
"""
