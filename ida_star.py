from puzzle import PuzzleState, is_goal, h_manhattan

def _dfs(path, g, limit):
    """
    Pesquisa em profundidade com limite no valor f(n) = g(n) + h(n).

    Esta função corresponde à fase recursiva do IDA*,
    explorando os estados até que o custo estimado ultrapasse o limite atual.
    """

    node = path[-1]
    f = g + node.h

    # Se o custo estimado exceder o limite, devolve esse valor
    if f > limit:
        return f

    # Verifica se foi atingido o estado objetivo
    if is_goal(node.grid):
        return "FOUND"

    # Guarda o menor excesso de f(n) para definir o próximo limite
    min_excess = float("inf")

    # Expansão dos vizinhos
    for neighbor in node.get_neighbors():
        # Evita ciclos no caminho atual
        if neighbor not in path:
            path.append(neighbor)

            # Chamada recursiva com custo incrementado
            t = _dfs(path, g + 1, limit)

            if t == "FOUND":
                return "FOUND"

            min_excess = min(min_excess, t)
            path.pop()

    return min_excess


def solve_ida_star(initial_grid):
    """
    Algoritmo IDA* (Iterative Deepening A*).

    Combina a ideia de A* com pesquisa em profundidade iterativa,
    utilizando limites sucessivos no valor f(n).
    Reduz significativamente o consumo de memória face ao A*,
    mantendo a optimalidade da solução.
    """

    # Estado inicial com heurística de Manhattan
    start = PuzzleState(initial_grid, h_manhattan)

    # Limite inicial baseado na heurística
    limit = start.h

    # Caminho atual da pesquisa
    path = [start]

    while True:
        t = _dfs(path, 0, limit)

        # Solução encontrada
        if t == "FOUND":
            return path

        # Não existe solução
        if t == float("inf"):
            return None

        # Atualiza o limite para o menor excesso encontrado
        limit = t
