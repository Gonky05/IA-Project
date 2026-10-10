from puzzle import h_manhattan, h_linear_conflict, set_goal, is_solvable
from dfs import solve_dfs
from bfs import solve_bfs
from gbfs import solve_gbfs
from a_star import solve_astar
from ida_star import solve_ida_star
from reinforcement_learning import solve_rl
from utils import reconstruct_path, print_grid

# ==================================================
# ESTADO INICIAL DO PUZZLE
# ==================================================
# Configuração fixa usada em todas as procuras
PUZZLE = [
    [6, 14, 5, 4],
    [1, 9, 2, 8],
    [13, 0, 10, 7],
    [11, 15, 3, 12]
]

# ==================================================
# OBJETIVOS DISPONÍVEIS
# ==================================================
# Permite alternar dinamicamente entre dois estados objetivo
GOALS = {
    "1": ("Objetivo clássico (0 no fim)", [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 0]
    ]),
    "2": ("Objetivo alternativo (0 no início)", [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]
    ]),
}

# ====================================================
# LIMITES PARA AS PROCURAS (nº de caminhos explorados)
# ====================================================
MAX_DEPTH_DFS = 20
MAX_NODES_BFS = 50_000
MAX_NODES_GBFS = 100_000

# ==================================================
# CATÁLOGO DE PROCURAS
# ==================================================
# Cada entrada associa:
#  - nome da procura
#  - função que executa a procura
#  - desenvolvemos esta parte do código utilizando a função lambda
SEARCH_CATALOG = {
    "1": ("DFS (limitada)", lambda: solve_dfs(PUZZLE, max_depth=MAX_DEPTH_DFS)),
    "2": ("BFS (limitada)", lambda: solve_bfs(PUZZLE, max_nodes=MAX_NODES_BFS)),
    "3": ("GBFS (Manhattan)", lambda: solve_gbfs(PUZZLE, heuristic=h_manhattan, max_nodes=MAX_NODES_GBFS)),
    "4": ("A* (Linear Conflict)", lambda: solve_astar(PUZZLE, heuristic=h_linear_conflict)),
    "5": ("IDA* (Manhattan)", lambda: solve_ida_star(PUZZLE)),
}

# ==================================================
# MENUS INTERATIVOS
# ==================================================

def choose_goal():
    """Menu de seleção do objetivo."""
    while True:
        print("\nEscolha o objetivo:")
        for k, (name, _) in GOALS.items():
            print(f"{k} - {name}")
        print("0 - Sair")

        op = input("Opção: ").strip()

        if op == "0" or op in GOALS:
            return op

        print("\nTecla não definida. Por favor, escolha uma opção válida.")

def menu():
    """Menu principal do programa."""
    while True:
        print("\nMenu principal:")
        print("1 - Executar uma procura")
        print("2 - Comparar duas procuras")
        print("3 - Reinforcement Learning")
        print("4 - Mudar objetivo")
        print("0 - Sair")

        op = input("Opção: ").strip()

        if op in {"0", "1", "2", "3", "4"}:
            return op

        print("\nTecla não definida. Por favor, escolha uma opção válida.")

def classic_menu():
    """Menu de escolha das procuras clássicas."""
    while True:
        print("\nProcuras disponíveis:")
        for k, (name, _) in SEARCH_CATALOG.items():
            print(f"{k} - {name}")
        print("0 - Voltar")

        op = input("Opção: ").strip()

        if op == "0" or op in SEARCH_CATALOG:
            return op

        print("\nTecla não definida. Por favor, escolha uma opção válida.")

def after_menu():
    """Menu apresentado após a execução de uma procura."""
    while True:
        print("\nO que deseja fazer a seguir?")
        print("1 - Voltar ao menu principal")
        print("2 - Mudar objetivo")
        print("0 - Sair")

        op = input("Opção: ").strip()

        if op in {"0", "1", "2"}:
            return op

        print("\nTecla não definida. Por favor, escolha uma opção válida.")

# ==================================================
# EXECUÇÃO DAS PROCURAS CLÁSSICAS
# ==================================================

def run_classic(choice):
    """Executa uma procura clássica selecionada pelo utilizador."""
    name, solver = SEARCH_CATALOG[choice]
    print(f"\nA executar: {name}")

    # Verificação prévia de solvabilidade
    if not is_solvable(PUZZLE):
        print("\nEste puzzle não é solúvel para o objetivo selecionado.")
        return

    solution = solver()
    if solution is None:
        print("\nNão foi encontrada solução.")
        return

    # Reconstrução da sequência de movimentos
    moves = reconstruct_path(solution)

    print("\nEstado inicial:")
    print_grid(solution[0].grid)

    for i, mv in enumerate(moves, 1):
        print(f"\nMovimento {i}: {mv}")
        print_grid(solution[i].grid)

    print(f"\nNúmero mínimo de movimentos: {len(moves)}")

# ==================================================
# EXECUÇÃO DO REINFORCEMENT LEARNING
# ==================================================

def run_rl():
    """Executa o algoritmo de Q-learning."""
    print("\nA executar: Reinforcement Learning (Q-learning)")

    # Também aqui se verifica a solvabilidade
    if not is_solvable(PUZZLE):
        print("\nEste puzzle não é solúvel para o objetivo selecionado.")
        return

    result = solve_rl(
        PUZZLE,
        episodes=30000,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.2,
        max_steps=200
    )

    if result is None:
        print("\nO agente não conseguiu resolver o puzzle.")
        return

    path, moves = result

    print("\nEstado inicial:")
    print_grid(path[0])

    for i, mv in enumerate(moves, 1):
        print(f"\nMovimento {i}: {mv}")
        print_grid(path[i])

    print(f"\nNúmero total de movimentos: {len(moves)}")

# ==================================================
# LOOP PRINCIPAL DO PROGRAMA
# ==================================================

while True:
    # Escolha inicial do objetivo
    g = choose_goal()

    if g == "0":
        break

    goal_name, goal_grid = GOALS[g]
    set_goal(goal_grid)
    print(f"\nObjetivo selecionado: {goal_name}")

    # Menu principal dependente do objetivo escolhido
    while True:
        op = menu()

        if op == "0":
            raise SystemExit

        if op == "4":
            break  # voltar ao menu de objetivos

        if op == "1":
            c = classic_menu()
            if c != "0":
                run_classic(c)

            nxt = after_menu()
            if nxt == "0":
                raise SystemExit
            if nxt == "2":
                break

        elif op == "2":
            c1 = classic_menu()
            c2 = classic_menu()
            if c1 in SEARCH_CATALOG and c2 in SEARCH_CATALOG:
                print("\nPrimeira procura:")
                run_classic(c1)
                print("\nSegunda procura:")
                run_classic(c2)

            nxt = after_menu()
            if nxt == "0":
                raise SystemExit
            if nxt == "2":
                break

        elif op == "3":
            run_rl()
            nxt = after_menu()
            if nxt == "0":
                raise SystemExit
            if nxt == "2":
                break
