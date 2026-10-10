import random
import puzzle

# ==================================================
# DEFINIÇÃO DAS AÇÕES POSSÍVEIS
# ==================================================
# Cada ação corresponde ao movimento do espaço vazio
# (nome, deslocamento em linha, deslocamento em coluna)
ACTIONS = {
    0: ("Cima",    -1,  0),
    1: ("Baixo",    1,  0),
    2: ("Esquerda", 0, -1),
    3: ("Direita",  0,  1),
}

# Ações opostas (útil para evitar desfazer movimentos)
OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}


# ==================================================
# FUNÇÕES AUXILIARES DE REPRESENTAÇÃO
# ==================================================

def flatten(grid):
    """Converte uma matriz 4x4 num tuplo (estado)."""
    return tuple(v for row in grid for v in row)

def unflatten(state):
    """Converte um estado (tuplo) numa matriz 4x4."""
    return [list(state[i:i+4]) for i in range(0, 16, 4)]

def goal_state():
    """Estado objetivo atual (dependente do goal definido no puzzle)."""
    return flatten(puzzle.GOAL_GRID)

def is_goal(state):
    """Verifica se o estado corresponde ao objetivo."""
    return state == goal_state()


# ==================================================
# DINÂMICA DO PUZZLE
# ==================================================

def find_empty(state):
    """Devolve a posição (linha, coluna) do espaço vazio."""
    idx = state.index(0)
    return divmod(idx, 4)

def valid_actions(state):
    """Determina quais as ações válidas a partir de um estado."""
    r, c = find_empty(state)
    valid = [0, 0, 0, 0]

    for a, (_, dr, dc) in ACTIONS.items():
        nr, nc = r + dr, c + dc
        if 0 <= nr < 4 and 0 <= nc < 4:
            valid[a] = 1

    return valid

def apply_action(state, action):
    """Aplica uma ação válida e devolve o novo estado."""
    r, c = find_empty(state)
    _, dr, dc = ACTIONS[action]
    nr, nc = r + dr, c + dc

    zero_pos = r * 4 + c
    new_pos = nr * 4 + nc

    s = list(state)
    s[zero_pos], s[new_pos] = s[new_pos], s[zero_pos]
    return tuple(s)


# ==================================================
# HEURÍSTICA (APENAS PARA REWARD SHAPING)
# ==================================================

def manhattan(state):
    """Distância de Manhattan usada apenas para cálculo do reward."""
    d = 0
    for i, v in enumerate(state):
        if v != 0:
            r, c = divmod(i, 4)
            gr, gc = puzzle.GOAL_POSITIONS[v]
            d += abs(r - gr) + abs(c - gc)
    return d


# ==================================================
# GERAÇÃO DE ESTADOS DE TREINO
# ==================================================

def scramble_from_goal(steps=10):
    """
    Gera estados solúveis aplicando movimentos aleatórios
    a partir do estado objetivo.
    """
    s = goal_state()
    last = None

    for _ in range(steps):
        valid = valid_actions(s)
        acts = [a for a, ok in enumerate(valid) if ok]

        # Evita desfazer imediatamente o último movimento
        if last is not None and OPPOSITE[last] in acts and len(acts) > 1:
            acts.remove(OPPOSITE[last])

        a = random.choice(acts)
        s = apply_action(s, a)
        last = a

    return s


# ==================================================
# AMBIENTE DE APRENDIZAGEM
# ==================================================

class Env:
    """
    Ambiente do puzzle para Q-learning.
    Controla o estado atual e a aplicação das ações.
    """
    def __init__(self, initial_state):
        self.initial = initial_state
        self.state = initial_state

    def reset(self, state=None):
        """Reinicia o ambiente."""
        self.state = self.initial if state is None else state
        return self.state

    def step(self, action):
        """
        Executa uma ação e devolve:
        (novo estado, reward, terminou)
        """
        if not valid_actions(self.state)[action]:
            raise ValueError("Ação inválida")

        nxt = apply_action(self.state, action)

        # Reward shaping:
        # grande recompensa no objetivo, caso contrário progresso heurístico
        reward = 100 if is_goal(nxt) else (manhattan(self.state) - manhattan(nxt)) - 1

        self.state = nxt
        return nxt, reward, is_goal(nxt)


# ==================================================
# Q-LEARNING
# ==================================================

def solve_rl(
    initial_grid,
    episodes=30000,
    alpha=0.1,
    gamma=0.99,
    epsilon=0.2,
    max_steps=200,
    scramble_steps=10,
    start_from_initial_prob=0.1,
    seed=None
):
    """
    Resolve o Jogo dos 15 usando Q-learning.

    Retorna:
      - sequência de matrizes (path)
      - lista de movimentos
    ou None se não encontrar solução.
    """

    if seed is not None:
        random.seed(seed)

    initial_state = flatten(initial_grid)
    env = Env(initial_state)
    Q = {}

    def ensure(s):
        """Inicializa a tabela Q para um estado, se necessário."""
        if s not in Q:
            Q[s] = [0.0, 0.0, 0.0, 0.0]

    # ---------------- TREINO ----------------
    for _ in range(episodes):
        # Treina a partir do estado inicial ou de estados próximos do objetivo
        state = (
            env.reset(initial_state)
            if random.random() < start_from_initial_prob
            else env.reset(scramble_from_goal(scramble_steps))
        )

        for _ in range(max_steps):
            ensure(state)
            valid = valid_actions(state)
            possible = [a for a, ok in enumerate(valid) if ok]

            # Política ε-greedy
            if random.random() < epsilon:
                action = random.choice(possible)
            else:
                q_vals = [Q[state][a] if valid[a] else -float("inf") for a in range(4)]
                max_q = max(q_vals)
                best = [a for a, q in enumerate(q_vals) if q == max_q]
                action = random.choice(best)

            nxt, r, done = env.step(action)
            ensure(nxt)

            # Atualização da Q-table
            nxt_valid = valid_actions(nxt)
            future = max(Q[nxt][a] for a in range(4) if nxt_valid[a])

            Q[state][action] += alpha * (r + gamma * future - Q[state][action])
            state = nxt

            if done:
                break

    # ---------------- EXECUÇÃO DA POLÍTICA ----------------
    state = env.reset(initial_state)
    path = [state]
    moves = []

    for _ in range(max_steps):
        if is_goal(state):
            break

        valid = valid_actions(state)
        possible = [a for a, ok in enumerate(valid) if ok]

        if state not in Q:
            action = random.choice(possible)
        else:
            q_vals = [Q[state][a] if valid[a] else -float("inf") for a in range(4)]
            max_q = max(q_vals)
            best = [a for a, q in enumerate(q_vals) if q == max_q]
            action = random.choice(best)

        state, _, _ = env.step(action)
        moves.append(ACTIONS[action][0])
        path.append(state)

    if not is_goal(state):
        return None

    return [unflatten(s) for s in path], moves
