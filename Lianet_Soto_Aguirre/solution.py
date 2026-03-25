from player import Player
from board import HexBoard
import random
import time
import math
import heapq


class SmartPlayer(Player):
    def __init__(self, player_id: int):
        super().__init__(player_id)

    def play(self, board: HexBoard) -> tuple:
        """
        Decide la mejor jugada usando MCTS con RAVE y heurísticas.
        """
        start_time = time.time()
        time_limit = 4.5  # segundos

        # Si el tablero está vacío, jugar cerca del centro
        empty_cells = board.get_empty_cells()
        if not empty_cells:
            return (0, 0)

        # Intentar bloquear amenazas críticas del rival
        threat_blocks = find_threat_blocks(board, self.player_id)
        if threat_blocks:
            ordered_blocks = sorted(
                threat_blocks,
                key=lambda mv: -evaluate_block_move(board, mv, self.player_id)
            )
            return ordered_blocks[0]

        # Si el tablero está vacío, jugar cerca del centro
        if len(empty_cells) == board.size * board.size:
            center = (board.size // 2, board.size // 2)
            if center in empty_cells:
                return center

        # MCTS normal con RAVE
        root = MCTSNode(board, player_to_move=self.player_id,
                        move=None, parent=None, root_player=self.player_id)

        while time.time() - start_time < time_limit:
            # --- Selección ---
            node = root
            while node.is_fully_expanded() and not node.is_terminal():
                node = node.select_child()

            # --- Expansión o evaluación terminal ---
            if node.is_terminal():
                if node.board.check_connection(1):
                    winner = 1
                elif node.board.check_connection(2):
                    winner = 2
                else:
                    winner = 0
                moves_played = []
            else:
                node = node.expand()
                winner, moves_played = node.simulate()

            # --- Retropropagación con RAVE ---
            node.backpropagate(winner, moves_played)

        # Elegir el hijo con más visitas
        if not root.children:
            return random.choice(empty_cells)

        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move

class MCTSNode:
    C = 1.0       # parámetro de exploración UCB
    K_RAVE = 300  # parámetro de mezcla RAVE

    def __init__(self, board: HexBoard, player_to_move: int,
                 move: tuple | None, parent: "MCTSNode | None",
                 root_player: int):
        self.board = board
        self.player_to_move = player_to_move
        self.move = move
        self.parent = parent
        self.root_player = root_player

        self.children: list[MCTSNode] = []
        self.visits = 0
        self.wins = 0.0  # victorias vistas desde la perspectiva de root_player

        # RAVE stats: para movimientos legales desde este nodo
        self.rave_wins: dict[tuple, float] = {}
        self.rave_visits: dict[tuple, int] = {}

        # Movimientos aún no expandidos (ordenados heurísticamente)
        self.untried_moves = heuristic_order(board, player_to_move)
        random.shuffle(self.untried_moves)

    # --------- Métodos auxiliares de estado ---------

    def is_terminal(self) -> bool:
        return self.board.check_connection(1) or self.board.check_connection(2)

    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0

    # --------- Expansión ---------

    def expand(self) -> "MCTSNode":
        move = self.untried_moves.pop()
        new_board = self.board.clone()
        new_board.place_piece(move[0], move[1], self.player_to_move)
        next_player = 3 - self.player_to_move
        child = MCTSNode(new_board, next_player, move, self, self.root_player)
        self.children.append(child)
        return child

    # --------- Selección (UCB + RAVE) ---------

    def select_child(self) -> "MCTSNode":
        # Hijos no visitados (usar RAVE o aleatorio)
        unvisited = [c for c in self.children if c.visits == 0]
        if unvisited:
            best = None
            best_rave = -float("inf")
            for c in unvisited:
                mv = c.move
                rvis = self.rave_visits.get(mv, 0)
                if rvis > 0:
                    rrate = self.rave_wins.get(mv, 0.0) / rvis
                else:
                    rrate = 0.0
                if rrate > best_rave:
                    best_rave = rrate
                    best = c
            return best if best is not None else random.choice(unvisited)

        # Todos tienen visitas (UCB+RAVE combinado)
        log_parent = math.log(self.visits + 1e-9)
        best = None
        best_score = -float("inf")

        for child in self.children:
            mv = child.move

            # Estadística normal
            win_rate = child.wins / child.visits if child.visits > 0 else 0.0

            # Estadística RAVE desde este nodo
            rave_vis = self.rave_visits.get(mv, 0)
            if rave_vis > 0:
                rave_rate = self.rave_wins.get(mv, 0.0) / rave_vis
            else:
                rave_rate = 0.0

            # Mezcla RAVE
            beta = math.sqrt(self.K_RAVE / (3 * child.visits + self.K_RAVE))
            combined_value = (1 - beta) * win_rate + beta * rave_rate

            ucb = combined_value + self.C * math.sqrt(
                log_parent / (child.visits + 1e-9)
            )

            if ucb > best_score:
                best_score = ucb
                best = child

        return best

    # --------- Simulación (playout sesgado + evaluación) ---------

    def simulate(self) -> tuple[int, list[tuple]]:
        sim_board = self.board.clone()
        player = self.player_to_move
        moves_played: list[tuple] = []

        max_steps = sim_board.size * sim_board.size

        for _ in range(max_steps):
            if sim_board.check_connection(1):
                return 1, moves_played
            if sim_board.check_connection(2):
                return 2, moves_played

            empty = sim_board.get_empty_cells()
            if not empty:
                break

            ordered = heuristic_order(sim_board, player)
            if not ordered:
                break

            # Sesgo: 70% mejores 3, 30% aleatorio
            if random.random() < 0.7:
                top_k = ordered[:min(3, len(ordered))]
                move = random.choice(top_k)
            else:
                move = random.choice(ordered)

            sim_board.place_piece(move[0], move[1], player)
            moves_played.append(move)
            player = 3 - player

        # Nadie ganó, usar evaluación heurística
        score = evaluate_board(sim_board, self.root_player)
        if score > 0:
            return self.root_player, moves_played
        elif score < 0:
            return 3 - self.root_player, moves_played
        else:
            return 0, moves_played

    # --------- Retropropagación con RAVE ---------

    def backpropagate(self, winner: int, moves_played: list[tuple]) -> None:
        node: MCTSNode | None = self
        seen_moves: set[tuple] = set()

        while node is not None:
            node.visits += 1

            # Actualizar victoria desde la perspectiva del root_player
            if winner == node.root_player:
                node.wins += 1.0
            elif winner == 0:
                node.wins += 0.5  # empate (por si acaso)

            # Actualizar RAVE: todos los movimientos jugados despues del movimiento de este nodo
            for mv in moves_played:
                if mv in seen_moves:
                    continue
                node.rave_visits[mv] = node.rave_visits.get(mv, 0) + 1
                if winner == node.player_to_move:
                    node.rave_wins[mv] = node.rave_wins.get(mv, 0.0) + 1.0
            seen_moves.update(moves_played)

            node = node.parent


# ===================== HEURÍSTICAS =====================

def heuristic_order(board: HexBoard, player_id: int) -> list[tuple]:
    """
    Ordena las casillas vacías por:
    - cercanía al centro,
    - progreso en la dirección objetivo,
    - proximidad a grupos propios,
    - bloqueo potencial al rival.
    """
    empty = board.get_empty_cells()
    if not empty:
        return []

    n = board.size
    center_r = (n - 1) / 2.0
    center_c = (n - 1) / 2.0

    opponent = 3 - player_id
    my_cells = [(r, c) for r in range(n) for c in range(n)
                if board.board[r][c] == player_id]
    opp_cells = [(r, c) for r in range(n) for c in range(n)
                 if board.board[r][c] == opponent]

    def dist(a: tuple, b: tuple) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    scored = []
    for (r, c) in empty:
        # centralidad
        d_center = abs(r - center_r) + abs(c - center_c)
        score = -d_center

        # progreso en la dirección objetivo
        if player_id == 1:  
            score += 0.3 * c
        else:
            score += 0.3 * r

        # cercanía a piezas propias
        if my_cells:
            best_my = min(dist((r, c), m) for m in my_cells)
            score += max(0, 3 - best_my)  # bonus si está cerca

        # bloqueo al rival
        if opp_cells:
            best_opp = min(dist((r, c), o) for o in opp_cells)
            score += max(0, 2 - best_opp)  # pequeño bonus defensivo

        scored.append(((r, c), score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [mv for mv, _ in scored]


def find_threat_blocks(board: HexBoard, defender_id: int) -> list[tuple]:
    """
    Devuelve casillas vacías que bloquean caminos cortos del rival.
    defender_id: jugador que va a mover (id propio).
    """
    attacker = 3 - defender_id
    empty = board.get_empty_cells()
    if not empty:
        return []

    base_cost = estimate_connection_cost(board, attacker)

    critical_moves = []
    for (r, c) in empty:
        tmp = board.clone()
        tmp.place_piece(r, c, attacker)
        new_cost = estimate_connection_cost(tmp, attacker)
        # Si el atacante mejoraría mucho su conexión jugando ahí,
        # entonces es una amenaza que conviene bloquear.
        if new_cost < base_cost - 3:
            critical_moves.append((r, c))

    return critical_moves


def evaluate_block_move(board: HexBoard, move: tuple, player_id: int) -> float:
    """
    Evalúa qué tan buena es una casilla de bloqueo:
    - cuánto empeora el camino del rival,
    - cuánto mejora el nuestro.
    """
    opp = 3 - player_id
    r, c = move
    tmp = board.clone()
    tmp.place_piece(r, c, player_id)

    opp_before = estimate_connection_cost(board, opp)
    opp_after = estimate_connection_cost(tmp, opp)
    my_before = estimate_connection_cost(board, player_id)
    my_after = estimate_connection_cost(tmp, player_id)

    delta_opp = opp_after - opp_before     # cuanto empeora su camino
    delta_me = my_before - my_after        # cuanto mejora el mío

    return 2.0 * delta_opp + 1.0 * delta_me


# ===================== EVALUACIÓN HEURÍSTICA =====================

def evaluate_board(board: HexBoard, player_id: int) -> float:
    """
    >0 favorece a player_id, <0 favorece al rival.
    Combina:
    - costo de conexión (ofensivo/defensivo),
    - control de camino propio (cantidad de fichas alineadas).
    """
    if board.check_connection(player_id):
        return 1e9
    opp = 3 - player_id
    if board.check_connection(opp):
        return -1e9

    my_cost = estimate_connection_cost(board, player_id)
    opp_cost = estimate_connection_cost(board, opp)

    my_chain_score = path_influence_score(board, player_id)
    opp_chain_score = path_influence_score(board, opp)

    return (opp_cost - my_cost) + 0.5 * (my_chain_score - opp_chain_score)


def path_influence_score(board: HexBoard, player_id: int) -> int:
    """
    Mide cuántas fichas propias están razonablemente cerca
    de formar caminos.
    """
    n = board.size
    score = 0
    if player_id == 1:
        # filas con continuidad aproximada izquierda-derecha
        for r in range(n):
            row_cells = [board.board[r][c] for c in range(n)]
            run = 0
            for v in row_cells:
                if v == player_id:
                    run += 1
                    if run >= 2:
                        score += run
                else:
                    run = 0
    else:
        # columnas con continuidad aproximada arriba-abajo
        for c in range(n):
            col_cells = [board.board[r][c] for r in range(n)]
            run = 0
            for v in col_cells:
                if v == player_id:
                    run += 1
                    if run >= 2:
                        score += run
                else:
                    run = 0
    return score


def estimate_connection_cost(board: HexBoard, player_id: int) -> int:
    """
    Estima el costo mínimo para conectar los lados requeridos.
    Para jugador 1: lados izquierdo-derecho.
    Para jugador 2: lados superior-inferior.
    """

    n = board.size
    opp = 3 - player_id

    # coste por tipo de casilla
    # propia: muy barato, vacía: 1, rival: 5
    def cell_cost(r: int, c: int) -> int:
        v = board.board[r][c]
        if v == player_id:
            return 0
        if v == 0:
            return 1
        if v == opp:
            return 5
        return 1

    def neighbors(r: int, c: int):
        return board._neighbors(r, c)

    # Dijkstra desde un lado al otro
    dist = [[float("inf")] * n for _ in range(n)]
    heap: list[tuple[float, int, int]] = []

    if player_id == 1:
        # conectar izquierda (col 0) con derecha (col n-1)
        for r in range(n):
            cost = cell_cost(r, 0)
            dist[r][0] = cost
            heapq.heappush(heap, (cost, r, 0))
    else:
        # conectar arriba (fila 0) con abajo (fila n-1)
        for c in range(n):
            cost = cell_cost(0, c)
            dist[0][c] = cost
            heapq.heappush(heap, (cost, 0, c))

    while heap:
        d, r, c = heapq.heappop(heap)
        if d > dist[r][c]:
            continue

        if player_id == 1 and c == n - 1:
            return int(d)
        if player_id == 2 and r == n - 1:
            return int(d)

        for nr, nc in neighbors(r, c):
            nd = d + cell_cost(nr, nc)
            if nd < dist[nr][nc]:
                dist[nr][nc] = nd
                heapq.heappush(heap, (nd, nr, nc))

    # Si no encontramos camino, devolver algo grande
    return n * n * 10