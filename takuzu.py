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

    # TODO: outros metodos da classe


class Board:
    """Representação interna de um tabuleiro de Takuzu."""
    size = 0
    board = []

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
        ct0, ct1 = 0, 0
        for y in range(self.size):
            if self.board[row][y] == 0:
                ct0 += 1
            if self.board[row][y] == 1:
                ct1 += 1
        return (ct0, ct1)

    def count_column(self, col: int) -> (int, int):
        ct0, ct1 = 0, 0
        for x in range(self.size):
            if self.board[x][col] == 0:
                ct0 += 1
            if self.board[x][col] == 1:
                ct1 += 1
        return (ct0, ct1)

    def check_3_straight_vertical(self, row: int, col: int) -> bool:
        if row == 0 and row == self.size - 1:
            return False
        elif row == 0:
            return self.adjacent_vertical_numbers(row+1, col) == (val, val)
        elif row == self.size - 1:
            return self.adjacent_vertical_numbers(row-1, col) == (val, val)
        else:
            return self.adjacent_vertical_numbers(row-1, col) == (val, val) \
                or self.adjacent_vertical_numbers(row, col) == (val, val) \
                or self.adjacent_vertical_numbers(row+1, col) == (val, val)

    def check_3_straight_horizontal(self, row: int, col: int) -> bool:
        if col == 0 and col == self.size - 1:
            return False
        elif col == 0:
            return self.adjacent_horizontal_numbers(row, col+1) == (val, val)
        elif col == self.size - 1:
            return self.adjacent_horizontal_numbers(row, col-1) == (val, val)
        else:
            return self.adjacent_horizontal_numbers(row, col-1) == (val, val) \
                or self.adjacent_horizontal_numbers(row, col) == (val, val) \
                or self.adjacent_horizontal_numbers(row, col+1) == (val, val)

    def check_3_straight(self, row: int, col: int) -> bool:
        return self.check_3_straight_horizontal(row, col) \
            or self.check_3_straight_vertical(row, col)

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
    
    def __str__(self) -> str:
        return "\n".join(["\t".join(map(str, row)) for row in self.board])

    # TODO: outros metodos da classe


class Takuzu(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        # TODO
        pass

    def actions(self, state: TakuzuState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        res = []
        for x in range(state.size):
            for y in range(state.size):
                if state.board[x][y] == 2:
                    res.append((x,y,0))
                    res.append((x,y,1))
        return res

    def result(self, state: TakuzuState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        x, y, val = action
        new_board = state.board
        new_board[x][y] = val
        return TakuzuState(new_board)

    def goal_test(self, state: TakuzuState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas com uma sequência de números adjacentes."""
        # TODO
        pass

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass

    # TODO: outros metodos da classe

    def h(node: Node, action):
        def impossible(node: Node, action):
            state = node.state
            row, col, val = action
            return board.count_column(col)[val] == ceil(state.size / 2) or \
                    board.count_row(row)[val] == ceil(state.size / 2) or \
                    result(node.state, action).board.check_3_straight(row, col)

        def mandatory(node: Node, action):
            conj_action = (action[0], action[1], 1-action[2])
            return not self.impossible(node, conj_action)

        def adjacency_tendency_row(node: Node, action):
            row, col, val = action
            board = node.state.board
            (x,y) = board.adjacent_horizontal_numbers(row, col)
            if (x,y) == (2,2) or (x,y) == (0,1) or (x,y) == (1,0):
                return 1/2
            elif (x,y) == (1,2) or (x,y) == (2,1):
                return 1
            elif (x,y) == (0,2) or (x,y) == (2,0):
                return 1

        if mandatory(node, action):
            return 0
        elif impossible(node, action):
            return 1

if __name__ == "__main__":
    # TODO:
    # Ler o ficheiro do standard input,
    board = Board.parse_instance_from_stdin()
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.
    pass
