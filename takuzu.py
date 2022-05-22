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

    def __init__(self, board):
        self.board = board
        self.id = TakuzuState.state_id
        TakuzuState.state_id += 1

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
            val = self.board[row][y]
            if val in (0, 1):
                counts[val] += 1
        return tuple(counts)

    def count_column(self, col: int) -> (int, int):
        counts = [0, 0]
        for x in range(self.size):
            val = self.board[x][col]
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
        
        vertical_adjacencies = [self.adjacent_vertical_numbers(row + i, col) for i in range(-1, 1)]
        horizontal_adjacencies = [self.adjacent_horizontal_numbers(row, col + i) for i in range(-1, 1)]
        return checker(row, vertical_adjacencies) or checker(col, horizontal_adjacencies)

    def cell_empty(self, row, col):
        return self.board[row][col] == self.EMPTY_CELL

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

    def actions(self, state: TakuzuState):
        moves = []
        board = state.board
        # in the beginning of the moves list will always be the mandatory moves
        for x in range(board.size):
            for y in range(board.size):
                if board.cell_empty(x, y):
                    possible_moves = [(x, y, 0), (x, y, 1)]
                    if self.mandatory(state, possible_moves[0]):
                        moves.insert(0, possible_moves[0])
                    elif not self.impossible(state, possible_moves[0]):
                        moves.append(possible_moves[0])
                    
                    if self.mandatory(state, possible_moves[1]):
                        moves.insert(0, possible_moves[1])
                    elif not self.impossible(state, possible_moves[1]):
                        moves.append(possible_moves[1])                    
        return moves

    def result(self, state: TakuzuState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        x, y, val = action
        updated_board = state.board.copy()
        updated_board.fill_cell(x, y, val)
        return TakuzuState(updated_board)

    def goal_test(self, state: TakuzuState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas com uma sequência de números adjacentes."""
        board = state.board
        rows = [tuple(row) for row in board.board]
        columns = [tuple(column) for column in board.transpose_board()]
        # and check if the board has an equal number of 0's and 1's (or off by 1, in case of odd sizes)
        return len(rows) == len(set(rows)) and \
            len(columns) == len(set(columns))  and \
            all(board.count_row(i) == (board.size, board.size) for i in range(board.size)) and \
            all(board.count_column(i) == (board.size, board.size) for i in range(board.size))
            
    def h(self, node: Node):
        # def adjacency_tendency_row(node: Node, action):
        #     row, col, val = action
        #     board = node.state.board
        #     (x,y) = board.adjacent_horizontal_numbers(row, col)
        #     if (x,y) == (2,2) or (x,y) == (0,1) or (x,y) == (1,0):
        #         return 1/2
        #     elif (x,y) == (1,2) or (x,y) == (2,1):
        #         return 1
        #     elif (x,y) == (0,2) or (x,y) == (2,0):
        #         return 1

        # moves = self.actions(node.state)
        # if self.mandatory(node, moves[0]):
        #     return 0
        # elif self.impossible(node, moves[0]):
        #     return 1
        return 1
    
    def impossible(self, node: Node, action):
        """Verifica se executar a ação-argumento leva a um estado em que é
        impossível completar o tabuleiro segundo as regras do jogo."""
        state = node.state
        hyp_state = self.result(state, action)
        row, col, val = action
        cap = int(state.size // 2 + bool(state.size % 2)) # produces ceiling
        return board.count_column(col)[val] == cap or \
                board.count_row(row)[val] == cap or \
                hyp_state.board.check_3_straight(row, col)

    def mandatory(self, node: Node, action):
        """Verifica se executar a ação-argumento é obrigatório - ou seja,
        se não é possível colocar outro valor na posição pretendida."""
        conj_action = (action[0], action[1], 1-action[2])
        # antes estava _not_ impossible, mas tipo, só é obrigatório se
        # o inverso for impossivel, certo?
        return self.impossible(node, conj_action)

if __name__ == "__main__":
    board = Board.parse_instance_from_stdin()
    # print(board)
    takuzu = Takuzu(board)
    # obviamente depois a estratégia varia, e temos de testar várias
    goal = recursive_best_first_search(takuzu)
    print(goal.state.board)
