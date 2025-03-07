"""
Implementation of Monte-Carlo tree search for AC graph.

Example:
Trivialize Akbulut-Kirby series n=2 case "AK(2)" through greedy search as 
python greedy.py
"""

import sys
import numpy as np
import heapq

sys.path.insert(0, "C:\\Users\\this_\\Documents\\GitHub\\AC-Solver")
from ac_solver.envs.utils import is_presentation_trivial
from ac_solver.envs.ac_moves import ACMove

def mcts(
    presentation,
    max_nodes_to_explore=10000,
    verbose=False,
    cyclically_reduce_after_moves=False,
):
    """
    Performs a Monte-Carlo tree search on an AC graph starting from the given presentation.
    """

    presentation = np.array(
        presentation, dtype=np.int8
    )  # so that input may be a list or a tuple

    # set initial state for search and maximum relator length allowed
    # if we encounter a presentation with a relator of length greater than max_relator_length,
    initial_state = np.array(
        presentation, dtype=np.int8
    )  # so that input may be a list or a tuple
    max_relator_length = len(presentation) // 2

    # we keep track of word lengths
    first_word_length = np.count_nonzero(presentation[:max_relator_length])
    second_word_length = np.count_nonzero(presentation[max_relator_length:])
    word_lengths = [first_word_length, second_word_length]
    total_initial_length = sum(word_lengths)

    # TODO

# Define Node class for the search tree
class Node:
    def __init__(self, state, parent=None, move=None):
        self.state = np.copy(state)
        # Potentially, parent should be a list? The AC "graph" is not a tree.
        self.parent = parent
        self.children = {}  # Map of moves to child nodes
        self.visits = 0
        self.value = 0
        
        # untried moves should remove moves which undo the last move
        # Should also try to keep track of if we have seen this state already.
        self.untried_moves = get_possible_moves(state, move) 

        # Probably don't need this. Once we have found a terminal
        # state we are done.
        self.is_terminal = is_presentation_trivial(state)
        
    def uct_select_child(self, exploration_weight=1.0):
        # UCT formula: value/visits + exploration_weight * sqrt(log(parent visits) / visits)
        log_parent_visits = np.log(self.visits)
        
        def uct_score(child):
            exploit = child.value / child.visits if child.visits > 0 else 0
            explore = exploration_weight * np.sqrt(log_parent_visits / child.visits) if child.visits > 0 else float('inf')
            return exploit + explore
            
        return max(self.children.values(), key=uct_score)
        
    def expand(self):
        # Pick an untried move and add a new child node
        move = self.untried_moves.pop()

        # Need to get the write syntax to call this.
        new_state = ACMove(self.state, move)

        # Make a set of seen moves?
        child = Node(new_state, parent=self, move=move)
        self.children[move] = child
        return child
        
    def is_fully_expanded(self):
        return len(self.untried_moves) == 0
        
    def backpropagate(self, result):
        self.visits += 1
        self.value += result
        if self.parent:
            self.parent.backpropagate(result)

def mcts(
    initial_state, max_nodes_to_explore=1600, verbose=True):
    # Main MCTS algorithm
    root = Node(initial_state)
    nodes_explored = 0
    
    while nodes_explored < max_nodes_to_explore:
        # Selection
        node = root
        while node.is_fully_expanded() and not node.is_terminal:
            node = node.uct_select_child()
            
        # Expansion
        if not node.is_terminal:
            node = node.expand()
            nodes_explored += 1
            
        # Simulation
        # I guess this should be the length of the presentation?
        result = simulate(node.state)
            
        # Backpropagation
        node.backpropagate(result)
        
        # Check if we found a solution
        if node.is_terminal:
            if verbose:
                print(f"Found solution after exploring {nodes_explored} nodes!")
            break
    
    # Reconstruct solution path
    if verbose:
        print(f"Explored {nodes_explored} nodes")
        
    # Find best node
    best_node = max(root.children.values(), key=lambda n: n.value / n.visits if n.visits > 0 else 0) if root.children else None