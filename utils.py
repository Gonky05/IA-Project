def reconstruct_path(states):
    """
    Reconstrói a sequência de movimentos a partir de um caminho
    de estados consecutivos (PuzzleState).

    Cada movimento é determinado comparando a posição do espaço
    vazio entre dois estados sucessivos.
    """
    moves = []

    for a, b in zip(states, states[1:]):
        ra, ca = a.empty_pos
        rb, cb = b.empty_pos

        if rb - ra == 1:
            moves.append("Baixo")
        elif rb - ra == -1:
            moves.append("Cima")
        elif cb - ca == 1:
            moves.append("Direita")
        elif cb - ca == -1:
            moves.append("Esquerda")

    return moves


def print_grid(grid):
    """
    Imprime uma matriz 4×4 de forma legível no terminal,
    facilitando a visualização do estado atual do puzzle.
    """
    print("-" * 21)
    for row in grid:
        print(" ", " ".join(f"{v:2}" for v in row))
