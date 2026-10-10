from puzzle import PuzzleState, is_goal

def solve_dfs(initial_grid, max_depth=20):
    """
    Procura em profundidade (DFS) com limite de profundidade.

    Explora o espaço de estados seguindo um caminho até ao limite
    máximo definido, evitando ciclos ao não repetir estados
    já presentes no caminho atual.
    """
    start = PuzzleState(initial_grid)

    # Pilha: (estado atual, caminho até ao estado, profundidade)
    stack = [(start, [start], 0)]

    while stack:
        state, path, depth = stack.pop()

        # Verifica se o estado objetivo foi atingido
        if is_goal(state.grid):
            return path

        # Impede exploração para além da profundidade máxima
        if depth >= max_depth:
            continue

        # Expande vizinhos evitando ciclos no caminho atual
        for neighbor in state.get_neighbors():
            if neighbor not in path:
                stack.append((neighbor, path + [neighbor], depth + 1))

    # Nenhuma solução encontrada dentro do limite imposto
    return None
