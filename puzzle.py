import copy

# ==================================================
# DEFINIÇÃO DO ESTADO OBJETIVO (GOAL) – DINÂMICO
# ==================================================
# O estado objetivo pode ser alterado em tempo de execução,
# permitindo testar diferentes configurações finais do puzzle.

GOAL_GRID = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 0]
]

# Mapeamento auxiliar: peça -> posição no objetivo
# Usado para acelerar o cálculo das heurísticas
GOAL_POSITIONS = {
    v: (r, c)
    for r, row in enumerate(GOAL_GRID)
    for c, v in enumerate(row)
}


def set_goal(new_goal_grid):
    """
    Atualiza dinamicamente o estado objetivo e as posições das peças.
    Todas as heurísticas e testes passam a usar este novo objetivo.
    """
    global GOAL_GRID, GOAL_POSITIONS
    GOAL_GRID = new_goal_grid
    GOAL_POSITIONS = {
        v: (r, c)
        for r, row in enumerate(GOAL_GRID)
        for c, v in enumerate(row)
    }


def is_goal(grid):
    """Verifica se uma matriz corresponde ao estado objetivo atual."""
    return grid == GOAL_GRID


# ==================================================
# TESTE DE SOLVABILIDADE (PUZZLE 4×4)
# ==================================================
# Para matrizes de dimensão par (4×4), a solvabilidade depende:
#  - do número de inversões
#  - da linha do espaço vazio contada a partir da base
# A paridade do estado inicial tem de coincidir com a do objetivo.


def _inversion_count(grid):
    """Conta o número de inversões da matriz (ignorando o 0)."""
    flat = [v for row in grid for v in row if v != 0]
    inv = 0
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            if flat[i] > flat[j]:
                inv += 1
    return inv


def _blank_row_from_bottom(grid):
    """
    Devolve a posição da linha do espaço vazio,
    contada a partir da base da matriz (1 a 4).
    """
    for r in range(4):
        if 0 in grid[r]:
            return 4 - r
    raise ValueError("Matriz inválida: não contém o espaço vazio.")


def is_solvable(initial_grid):
    """
    Verifica se o estado inicial é solúvel para o objetivo atual,
    comparando a paridade (inversões + linha do vazio) de ambos.
    """
    inv_init = _inversion_count(initial_grid)
    inv_goal = _inversion_count(GOAL_GRID)

    blank_init = _blank_row_from_bottom(initial_grid)
    blank_goal = _blank_row_from_bottom(GOAL_GRID)

    return ((inv_init + blank_init) % 2) == ((inv_goal + blank_goal) % 2)


# ==================================================
# HEURÍSTICAS
# ==================================================
# Heurísticas informadas que estimam o custo até ao objetivo.


def h_manhattan(grid):
    """
    Heurística da distância de Manhattan:
    soma das distâncias verticais e horizontais de cada peça
    até à sua posição no estado objetivo.
    """
    d = 0
    for r in range(4):
        for c in range(4):
            v = grid[r][c]
            if v != 0:
                gr, gc = GOAL_POSITIONS[v]
                d += abs(r - gr) + abs(c - gc)
    return d


def h_linear_conflict(grid):
    """
    Heurística Conflito Linear:
    Extensão da Manhattan que adiciona penalizações
    quando duas peças estão na mesma linha ou coluna
    mas em ordem inversa relativamente ao objetivo.
    """
    h = h_manhattan(grid)

    # Conflitos em linhas
    for r in range(4):
        row = grid[r]
        for i in range(4):
            for j in range(i + 1, 4):
                a, b = row[i], row[j]
                if a != 0 and b != 0:
                    ga_r, ga_c = GOAL_POSITIONS[a]
                    gb_r, gb_c = GOAL_POSITIONS[b]
                    if ga_r == gb_r == r and ga_c > gb_c:
                        h += 2

    # Conflitos em colunas
    for c in range(4):
        col = [grid[r][c] for r in range(4)]
        for i in range(4):
            for j in range(i + 1, 4):
                a, b = col[i], col[j]
                if a != 0 and b != 0:
                    ga_r, ga_c = GOAL_POSITIONS[a]
                    gb_r, gb_c = GOAL_POSITIONS[b]
                    if ga_c == gb_c == c and ga_r > gb_r:
                        h += 2

    return h


# ==================================================
# CLASSE PuzzleState
# ==================================================
# Representa um nó no espaço de estados, encapsulando:
#  - a matriz atual
#  - a posição do espaço vazio
#  - o valor heurístico (se aplicável)

class PuzzleState:
    def __init__(self, grid, heuristic=None):
        self.grid = grid
        self.heuristic = heuristic
        self.empty_pos = self._find_empty()
        self.h = heuristic(grid) if heuristic else 0

    def _find_empty(self):
        """Localiza a posição do espaço vazio na matriz."""
        for r in range(4):
            for c in range(4):
                if self.grid[r][c] == 0:
                    return r, c
        raise ValueError("Matriz inválida: não contém o espaço vazio.")

    def get_neighbors(self):
        """
        Gera todos os estados vizinhos válidos,
        resultantes do movimento do espaço vazio.
        """
        neighbors = []
        r, c = self.empty_pos

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 4 and 0 <= nc < 4:
                g = copy.deepcopy(self.grid)
                g[r][c], g[nr][nc] = g[nr][nc], 0
                neighbors.append(PuzzleState(g, self.heuristic))

        return neighbors

    def __eq__(self, other):
        """Permite comparar estados pelo conteúdo da matriz."""
        return isinstance(other, PuzzleState) and self.grid == other.grid

    def __hash__(self):
        """Permite usar estados em estruturas como sets e dicionários."""
        return hash(tuple(tuple(row) for row in self.grid))
