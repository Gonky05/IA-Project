from collections import deque
from puzzle import PuzzleState, is_goal

def solve_bfs(initial_grid, max_nodes=50_000):
    """
    Procura em largura (BFS) com limite no número de nós expandidos.

    Explora o espaço de estados nível a nível, garantindo que a
    primeira solução encontrada é ótima em número de movimentos,
    desde que o limite de nós não seja atingido.
    """
    start = PuzzleState(initial_grid)

    # Fila: (estado atual, caminho até ao estado)
    queue = deque([(start, [start])])

    # Conjunto de estados já visitados (evita ciclos)
    visited = {start}

    # Contador de nós expandidos (custo)
    nodes_expanded = 0

    while queue:
        state, path = queue.popleft()
        nodes_expanded += 1

        # Verifica se o estado objetivo foi atingido
        if is_goal(state.grid):
            return path

        # Interrompe a procura se ultrapassar o limite definido
        if nodes_expanded >= max_nodes:
            return None

        # Expande todos os vizinhos do estado atual
        for neighbor in state.get_neighbors():
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    # Nenhuma solução encontrada dentro do limite imposto
    return None
