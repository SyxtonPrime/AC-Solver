"""
Implementation of Monte-Carlo tree search for AC graph.

Example:
Trivialize Akbulut-Kirby series n=2 case "AK(2)" through greedy search as 
python greedy.py
"""

import sys
import numpy as np
import heapq
from tqdm import tqdm

sys.path.insert(0, "C:\\Users\\this_\\Documents\\GitHub\\AC-Solver")
from ac_solver.envs.utils import is_presentation_trivial
from ac_solver.envs.ac_moves import ACMove

def monte_carlo_search(
    presentation,
    max_path_length=500,
    monte_carlo_simulations=5000,
    verbose=False,
    cyclically_reduce_after_moves=False,
):
    """
    Traverses the AC graph using a Monte-Carlo tree search algorithm.

    Parameters:
        presentation (np.ndarray): Initial presentation as a NumPy array.
        max_nodes_to_explore (int, optional): Max nodes to explore before termination (default: 10000).
        monte_carlo_simulations (int, optional): Number of Monte-Carlo simulations to run at each node (default: 20736 = 12**4).
        verbose (bool, optional): Print updates when shorter presentations are found (default: False).
        cyclically_reduce_after_moves (bool, optional): Apply cyclic reduction after each move (default: False).

    Returns:
        tuple: (is_search_successful, path)
            - is_search_successful (bool): Whether a trivial state was found.
            - path (list of tuple): Sequence of (action, presentation_length).
    """

    presentation = np.array(
        presentation, dtype=np.int8
    )  # so that input may be a list or a tuple

    # set initial state for search and maximum relator length allowed
    # if we encounter a presentation with a relator of length greater than max_relator_length,
    current_state = np.array(
        presentation, dtype=np.int8
    )  # so that input may be a list or a tuple
    max_relator_length = len(presentation) // 2

    # we keep track of word lengths
    first_word_length = np.count_nonzero(presentation[:max_relator_length])
    second_word_length = np.count_nonzero(presentation[max_relator_length:])
    word_lengths = [first_word_length, second_word_length]
    total_initial_length = sum(word_lengths)

    # a set containing states that have already been seen
    tree_nodes = set()
    tree_nodes.add(tuple(current_state))

    path = [(-1, total_initial_length)]
    current_root = Node(current_state, word_lengths)

    for _ in tqdm(range(max_path_length)):
        new_root, trivial_found = mcts(current_root, tree_nodes, max_relator_length, max_nodes_to_explore = monte_carlo_simulations)
        if trivial_found:
            print("Trivial found!")
            path += path_to_node(current_root, new_root)
            return True, path
        
        current_root = new_root
        path += [(int(new_root.move), int(new_root.word_lengths.sum()))]
        tree_nodes.add(tuple(new_root.state))
        current_state = new_root.state
        word_lengths = new_root.word_lengths
        
        

    return False, path

# Define Node class for the search tree
class Node:
    def __init__(self, state, word_lengths, parent=None, move=None):
        self.state = np.copy(state)
        self.word_lengths = np.copy(word_lengths)
        # Potentially, parent should be a list? The AC "graph" is not a tree.
        self.parent = parent
        # What move did we apply to the parent to get to this state?
        self.move = move
        self.children = {}  # Map of moves to child nodes
        self.visits = 0
        self.value = 0
        
        # TODO: untried moves should remove moves which undo the last move
        # For now, just shuffle all possible moves.
        self.untried_moves = list(np.random.permutation(12))
        
    def uct_select_child(self, exploration_weight=1.0):
        # UCT formula: value/visits + exploration_weight * sqrt(log(parent visits) / visits)
        log_parent_visits = np.log(self.visits)
        
        def uct_score(child):
            exploit = child.value / child.visits if child.visits > 0 else 0
            explore = exploration_weight * np.sqrt(log_parent_visits / child.visits) if child.visits > 0 else float('inf')
            return exploit + explore
            
        return max(self.children.values(), key=uct_score)
    
    def evaluate(self, max_relator_length):
        # Eventually this should be a neural network.
        return 2*max_relator_length - self.word_lengths.sum()
        
    def expand(self, previous_states, seen_nodes, max_relator_length):
        found_new_state = False

        while not (found_new_state or self.is_fully_expanded()):
             # Pick an untried move and add a new child node
            move = self.untried_moves.pop()

            # Need to get the write syntax to call this.
            new_state, lengths, found_new_state = ACMove(move, self.state, max_relator_length, self.word_lengths, cyclical=False)

            # Check if we have seen this state before
            if found_new_state and (tuple(new_state) in previous_states or tuple(new_state) in seen_nodes):
                found_new_state = False
        
        if not found_new_state:
            return None

        # Make a set of seen moves?
        child = Node(new_state, lengths, parent=self, move=move)
        self.children[move] = child
        return child
        
    def is_fully_expanded(self):
        return len(self.untried_moves) == 0
        
    def backpropagate(self, result):
        self.visits += 1
        self.value = max(self.value, result)
        if self.parent:
            self.parent.backpropagate(result)

def mcts(
    root, previous_states, max_relator_length, max_nodes_to_explore=1728, verbose=False):
    """"
        Traverses the AC graph using a Monte-Carlo tree search algorithm to find the best next step.

        Due to the nature of the problem, we don't want to revisit any state we have previously passed
        through.

        Parameters:
            initial_state (np.ndarray): Initial presentation as a NumPy array.
            previous_states (set): set of previous states.
            max_nodes_to_explore (int, optional): Max nodes to explore before termination (default: 1728).
            verbose (bool, optional): Print updates when shorter presentations are found (default: True).

        
    """
    # Main MCTS algorithm
    nodes_explored = 0
    seen_nodes = set()
    
    while nodes_explored < max_nodes_to_explore:
        # Selection
        node = root

        while node.is_fully_expanded():
            if len(node.children.values()) == 0:
                node.parent.children.pop(node.move)
                node = node.parent
            else:
                node = node.uct_select_child()
        
        # Expansion
        new_node = node.expand(previous_states, seen_nodes, max_relator_length)
        if new_node != None:
            node = new_node
            if is_presentation_trivial(node.state):
                # If we find a trivial presentation, we should it.
                return node, True

            seen_nodes.add(tuple(node.state))
            nodes_explored += 1
                
            # For now this the maximum length minus the length of the presentation.
            # Eventually this should be a neural net.
            result = node.evaluate(max_relator_length)
            node.value = result

            # Backpropagation
            node.backpropagate(result)

        
    # Find best node
    best_node = max(root.children.values(), key=lambda n: n.value / n.visits if n.visits > 0 else 0) if root.children else None

    return best_node, False

def path_to_node(current_root, new_root):
    path = [(int(new_root.move), int(new_root.word_lengths.sum()))]
    while new_root.parent != current_root:
        new_root = new_root.parent
        path += [(int(new_root.move), int(new_root.word_lengths.sum()))]
    
    path.reverse()
    return path

if __name__ == "__main__":

    # presentation = np.array([1, 1, -2, -2, -2, 0, 0, 0, 1, 2, 1, -2, -1, -2, 0, 0])  # AK(2)
    # presentation = np.array([1, 1, 1, -2, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, -2, -1, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # AK(3)
    # presentation = np.array([1, 0, 0, 0, 0, 2, 1, 1, 1, 0])

    # presentation = np.array([-1, 2, 2, 2, 1, -2, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, 1, -2, -1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    presentation = np.array([-1, 2, 1, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, 2, 1, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    ans, path = monte_carlo_search(presentation=presentation)

    if ans:
        print(
            f"""
              Presentation {presentation} solved!
              Path length: {len(path)}
              Path: {path} 
              """
        )
        print("Checking whether this path actually leads to a trivial state..")
        
        max_relator_length = len(presentation) // 2
        first_word_length = np.count_nonzero(presentation[:max_relator_length])
        second_word_length = np.count_nonzero(presentation[max_relator_length:])
        word_lengths = [first_word_length, second_word_length]

        for action, _ in path[1:]:
            presentation, word_lengths, _ = ACMove(
                move_id=action,
                presentation=presentation,
                max_relator_length=max_relator_length,
                lengths=word_lengths,
                cyclical=False,
            )

        print(f"Final state achieved: {presentation}")
        print(f"Is trivial? {is_presentation_trivial(presentation)}")
    else:
        print(
            f"""
              Presentation {presentation} not solved!
              """
              )