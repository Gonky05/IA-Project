import heapq
from itertools import count
from puzzle import PuzzleState, is_goal, h_linear_conflict

def solve_astar(initial_grid, heuristic=h_linear_conflict):
    """
    Algoritmo A* (A estrela).

    Combina o custo real g(n) desde o estado inicial com uma heurística
    admissível h(n), avaliando cada estado através de f(n) = g(n) + h(n).
    Garante encontrar uma solução ótima sempre que a heurística for admissível.
    """

    # Estado inicial
    start = PuzzleState(initial_grid, heuristic)

    # Contador para desempate na fila de prioridade
    tie = count()

    # Fila de prioridade: (f(n), g(n), desempate, estado, caminho)
    frontier = [(start.h, 0, next(tie), start, [start])]

    # Guarda o menor custo g(n) conhecido para cada estado
    g_cost = {start: 0}

    # Conjunto de estados já completamente explorados
    closed = set()

    while frontier:
        _, g, _, state, path = heapq.heappop(frontier)

        # Verifica se o estado atual é o objetivo
        if is_goal(state.grid):
            return path

        # Evita reexpandir estados já processados
        if state in closed:
            continue
        closed.add(state)

        # Expansão dos estados vizinhos
        for neighbor in state.get_neighbors():
            new_g = g + 1  # custo de mover para o vizinho

            # Atualiza se for encontrado um caminho mais barato
            if neighbor not in g_cost or new_g < g_cost[neighbor]:
                g_cost[neighbor] = new_g
                heapq.heappush(
                    frontier,
                    (new_g + neighbor.h, new_g, next(tie), neighbor, path + [neighbor])
                )

    # Caso não exista solução
    return None
