import heapq
from itertools import count
from puzzle import PuzzleState, is_goal, h_manhattan

def solve_gbfs(initial_grid, heuristic=h_manhattan, max_nodes=100_000):
    """
    Greedy Best-First Search (GBFS) com limite no número de nós expandidos.

    A procura é guiada exclusivamente pelo valor da heurística h(n),
    escolhendo sempre o estado que aparenta estar mais próximo do
    objetivo, sem garantir otimalidade.
    """
    start = PuzzleState(initial_grid, heuristic)

    # Contador auxiliar para desempate na fila de prioridade
    tie = count()

    # Fila de prioridade: (h(n), desempate, estado, caminho)
    frontier = [(start.h, next(tie), start, [start])]

    # Conjunto de estados já visitados
    visited = set()

    # Contador de nós expandidos
    nodes_expanded = 0

    while frontier:
        _, _, state, path = heapq.heappop(frontier)

        # Evita reprocessar estados já visitados
        if state in visited:
            continue

        visited.add(state)
        nodes_expanded += 1

        # Verifica se o objetivo foi atingido
        if is_goal(state.grid):
            return path

        # Interrompe a procura se o limite for atingido
        if nodes_expanded >= max_nodes:
            return None

        # Expande os vizinhos, ordenados implicitamente pela heurística
        for neighbor in state.get_neighbors():
            if neighbor not in visited:
                heapq.heappush(
                    frontier,
                    (neighbor.h, next(tie), neighbor, path + [neighbor])
                )

    # Nenhuma solução encontrada dentro do limite imposto
    return None
