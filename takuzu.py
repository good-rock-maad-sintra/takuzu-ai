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

    def adjacent_vertical_numbers(self, row: int, col: int) -> tuple(int, int):
        """Devolve os valores imediatamente abaixo e acima,
        respectivamente."""
        if not 0 <= row < self.size or not 0 <= row < self.size:
            return None
        return (self.get_number(row-1, col), self.get_number(row+1, col))

    def adjacent_horizontal_numbers(self, row: int, col: int) -> tuple(int, int):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        if not 0 <= col < self.size or not 0 <= col < self.size:
            return None
        return (self.get_number(row, col-1), self.get_number(row, col+1))

    def count_row(self, row: int) -> tuple(int, int):
        counts = [0, 0]
        for y in range(self.size):
            val = self.board[row][y]
            if val in (0, 1):
                counts[val] += 1
        return tuple(counts)

    def count_column(self, col: int) -> tuple(int, int):
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
        moves = []
        for x in range(state.size):
            for y in range(state.size):
                if board.cell_empty(x, y):
                    moves.append((x,y,0), (x,y,1))
        return moves

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

    def h(self, node: Node, action):
        def impossible(node: Node, action):
            """Verifica se executar a ação-argumento leva a um estado em que é
            impossível completar o tabuleiro segundo as regras do jogo."""
            state = node.state
            hyp_state = self.result(state, action)
            row, col, val = action
            cap = int(state.size // 2 + bool(state.size % 2)) # produces ceiling
            return board.count_column(col)[val] == cap or \
                    board.count_row(row)[val] == cap or \
                    hyp_state.board.check_3_straight(row, col)

        def mandatory(node: Node, action):
            """Verifica se executar a ação-argumento é obrigatório - ou seja,
            se não é possível colocar outro valor na posição pretendida."""
            conj_action = (action[0], action[1], 1-action[2])
            # antes estava _not_ impossible, mas tipo, só é obrigatório se
            # o inverso for impossivel, certo?
            return self.impossible(node, conj_action)

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
