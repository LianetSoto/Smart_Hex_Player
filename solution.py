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
        Decide la mejor jugada usando MCTS con límite de tiempo.
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
                # El juego ya terminó en este nodo
                winner = 1 if node.board.check_connection(1) else 2
            else:
                node = node.expand()
                winner = node.simulate()

            # --- Retropropagación ---
            node.backpropagate(winner, self.player_id)
            iterations += 1

        # Elegir el movimiento con mas visitas
        if not root.children:
            empty = board.get_empty_cells()
            return random.choice(empty) if empty else (0, 0)

        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move

    class MCTSNode:
        def __init__(self, board, player_to_move, move, parent):
            self.board = board
            self.player_to_move = player_to_move  # quien mueve desde este estado
            self.move = move  # movimiento que llevo a este nodo
            self.parent = parent
            self.children = []
            self.visits = 0
            self.wins = 0
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
            """Selecciona un hijo usando UCB1."""
            # Priorizar hijos no visitados
            unvisited = [c for c in self.children if c.visits == 0]
            if unvisited:
                return random.choice(unvisited)

            # Cálculo UCB para hijos visitados
            C = 1.4
            log_parent = math.log(self.visits)
            best = None
            best_ucb = -float('inf')
            for child in self.children:
                ucb = child.wins / child.visits + C * math.sqrt(log_parent / child.visits)
                if ucb > best_ucb:
                    best_ucb = ucb
                    best = child
            return best

        def simulate(self):
            """
            Realiza un playout aleatorio desde este estado hasta el final.
            Retorna el id del jugador ganador (1 o 2).
            """
            sim_board = self.board.clone()
            player = self.player_to_move
            while True:
                if sim_board.check_connection(1):
                    return 1
                if sim_board.check_connection(2):
                    return 2
                empty = sim_board.get_empty_cells()
                if not empty:
                    return 0
                move = random.choice(empty)
                sim_board.place_piece(move[0], move[1], player)
                player = 3 - player

        def backpropagate(self, winner, root_player):
            """Actualiza estadísticas a lo largo del camino hacia la raíz."""
            self.visits += 1
            if winner == root_player:
                self.wins += 1
            if self.parent:
                self.parent.backpropagate(winner, root_player)
