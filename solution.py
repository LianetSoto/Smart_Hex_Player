from player import Player
from board import HexBoard
import random
import time
import math


class SmartPlayer(Player):
    def __init__(self, player_id: int):
        super().__init__(player_id)

    def play(self, board: HexBoard) -> tuple:
        """
        Decide la mejor jugada usando MCTS con RAVE y límite de tiempo.
        """
        start_time = time.time()
        time_limit = 4.5  # segundos

        # Nodo raíz: el jugador actual mueve desde el estado dado
        root = self.MCTSNode(board, self.player_id, move=None, parent=None)

        iterations = 0
        while time.time() - start_time < time_limit:
            # --- Selección ---
            node = root
            while node.is_fully_expanded() and not node.is_terminal():
                node = node.select_child()

            # --- Expansión o evaluación terminal ---
            if node.is_terminal():
                winner = 1 if node.board.check_connection(1) else 2
                moves = []   # nodo terminal, no hay movimientos nuevos
            else:
                node = node.expand()
                winner, moves = node.simulate()

            # --- Retropropagación con RAVE ---
            node.backpropagate(winner, moves, root.player_to_move)
            iterations += 1

        # Elegir el movimiento con más visitas (o RAVE si no hay hijos)
        if not root.children:
            empty = board.get_empty_cells()
            return random.choice(empty) if empty else (0, 0)

        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move

    class MCTSNode:
        # Constantes para RAVE y exploración
        C = 1.4
        K_RAVE = 300

        def __init__(self, board, player_to_move, move, parent):
            self.board = board
            self.player_to_move = player_to_move  # quien mueve desde este estado
            self.move = move  # movimiento que llevó a este nodo
            self.parent = parent
            self.children = []
            self.visits = 0
            self.wins = 0
            # Estadísticas RAVE
            self.rave_wins = {}     # {movimiento: victorias RAVE}
            self.rave_visits = {}    # {movimiento: visitas RAVE}
            # Lista de movimientos no expandidos
            self.untried_moves = board.get_empty_cells()
            random.shuffle(self.untried_moves)

        def is_terminal(self):
            """True si el juego ha terminado."""
            return self.board.check_connection(1) or self.board.check_connection(2)

        def is_fully_expanded(self):
            """True si ya se expandieron todos los movimientos posibles."""
            return len(self.untried_moves) == 0

        def expand(self):
            """Crea un nuevo nodo hijo para el siguiente movimiento no probado."""
            move = self.untried_moves.pop()
            new_board = self.board.clone()
            new_board.place_piece(move[0], move[1], self.player_to_move)
            next_player = 3 - self.player_to_move  # alternar jugador
            child = SmartPlayer.MCTSNode(new_board, next_player, move, self)
            self.children.append(child)
            return child

        def select_child(self):
            """
            Selecciona un hijo usando UCB1 combinado con RAVE.
            Si hay hijos no visitados, elige el de mayor valor RAVE (si existe) o aleatorio.
            """
            # Hijos no visitados
            unvisited = [c for c in self.children if c.visits == 0]
            if unvisited:
                # Si hay RAVE para alguno, elegir el mejor; si no, aleatorio
                best = None
                best_rave = -float('inf')
                for c in unvisited:
                    rave_val = self.rave_wins.get(c.move, 0) / (self.rave_visits.get(c.move, 0) + 1e-5)
                    if rave_val > best_rave:
                        best_rave = rave_val
                        best = c
                if best is not None:
                    return best
                else:
                    return random.choice(unvisited)

            # Todos los hijos tienen visitas > 0 → usar UCB+RAVE
            log_parent = math.log(self.visits)
            best = None
            best_ucb = -float('inf')
            for child in self.children:
                move = child.move
                # Estadísticas normales
                win_rate = child.wins / child.visits
                # Estadísticas RAVE desde el nodo actual
                rave_vis = self.rave_visits.get(move, 0)
                rave_win = self.rave_wins.get(move, 0)
                rave_rate = rave_win / (rave_vis + 1e-5) if rave_vis > 0 else 0.0

                # Peso beta: depende de las visitas normales
                beta = math.sqrt(self.K_RAVE / (3 * child.visits + self.K_RAVE))
                # Valor combinado
                combined = (1 - beta) * win_rate + beta * rave_rate
                ucb = combined + self.C * math.sqrt(log_parent / child.visits)

                if ucb > best_ucb:
                    best_ucb = ucb
                    best = child
            return best

        def simulate(self):
            """
            Realiza un playout aleatorio desde este estado hasta el final.
            Retorna (winner, moves) donde moves es la lista de movimientos jugados.
            """
            sim_board = self.board.clone()
            player = self.player_to_move
            moves = []
            while True:
                if sim_board.check_connection(1):
                    return 1, moves
                if sim_board.check_connection(2):
                    return 2, moves
                empty = sim_board.get_empty_cells()
                if not empty:
                    return 0, moves   # empate
                move = random.choice(empty)
                moves.append(move)
                sim_board.place_piece(move[0], move[1], player)
                player = 3 - player

        def backpropagate(self, winner, moves, root_player):
            """
            Actualiza estadísticas (normales y RAVE) a lo largo del camino hacia la raíz.
            - winner: id del jugador ganador (1 o 2)
            - moves: lista de movimientos jugados desde este nodo (en orden)
            - root_player: id del jugador que movió primero en la raíz (para contar victorias)
            """
            self.visits += 1
            if winner == root_player:
                self.wins += 1

            # Actualizar RAVE para cada movimiento en 'moves'
            for move in moves:
                self.rave_visits[move] = self.rave_visits.get(move, 0) + 1
                if winner == self.player_to_move:
                    self.rave_wins[move] = self.rave_wins.get(move, 0) + 1

            # Propagar al padre (si existe)
            if self.parent:
                # Los movimientos desde el padre incluyen el que llevó a este nodo + los posteriores
                parent_moves = [self.move] + moves if self.move is not None else moves
                self.parent.backpropagate(winner, parent_moves, root_player)
