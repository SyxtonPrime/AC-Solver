"""
Implementation of greedy search for AC graph.

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


def greedy_search(
    presentation,
    max_nodes_to_explore=10000,
    n = 1,
    verbose=False,
    cyclically_reduce_after_moves=False,
):
    """
    Performs a greedy search on an AC graph starting from the given presentation.

    Parameters:
        presentation (np.ndarray): Initial presentation as a NumPy array.
        max_nodes_to_explore (int, optional): Max nodes to explore before termination (default: 10000).
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
    initial_state = np.array(
        presentation, dtype=np.int8
    )  # so that input may be a list or a tuple
    max_relator_length = len(presentation) // 2

    # we keep track of word lengths
    first_word_length = np.count_nonzero(presentation[:max_relator_length])
    second_word_length = np.count_nonzero(presentation[max_relator_length:])
    word_lengths = [first_word_length, second_word_length]
    total_initial_length = sum(word_lengths)

    # add to a priority queue, keeping track of path length to initial state
    path_length = 0
    to_explore = [
        (
            total_initial_length,
            path_length,
            tuple(initial_state),
            tuple(word_lengths),
            [(-1, total_initial_length)],
        )
    ]
    heapq.heapify(to_explore)

    # a set containing states that have already been seen
    tree_nodes = set()
    tree_nodes.add(tuple(initial_state))
    min_length = total_initial_length

    while to_explore:
        _, path_length, state_tuple, word_lengths, path = heapq.heappop(to_explore)
        state = np.array(state_tuple, dtype=np.int8)  # convert tuple to state
        word_lengths = list(word_lengths)

        exploration_state = State(state, word_lengths)

        trivial, path = exploration_state.propagate(n, path, tree_nodes, max_relator_length, to_explore)

        if trivial:
            return True, path

        # for action in range(0, 12):
        #     new_state, new_lengths, _ = ACMove(
        #         action,
        #         state,
        #         max_relator_length,
        #         word_lengths,
        #         cyclical=cyclically_reduce_after_moves,
        #     )
        #     state_tup, new_length = tuple(new_state), sum(new_lengths)

        #     if new_length < min_length:
        #         min_length = new_length
        #         if verbose:
        #             print(f"New minimal length found: {min_length}")

        #     if new_length == 2:
        #         if verbose:
        #             print(
        #                 f"Found {new_state[0:1], new_state[max_relator_length:max_relator_length+1]} after exploring {len(tree_nodes)-len(to_explore)} nodes"
        #             )
        #             print(
        #                 f"Path to a trivial state: (tuples are of form (action, length of a state)) {path + [(action, new_length)]}"
        #             )
        #             print(f"Total path length: {len(path)+1}")
        #         return True, path + [(action, new_length)]

        #     if state_tup not in tree_nodes:
        #         tree_nodes.add(state_tup)
        #         heapq.heappush(
        #             to_explore,
        #             (
        #                 new_length,
        #                 path_length + 1,
        #                 state_tup,
        #                 tuple(new_lengths),
        #                 path + [(action, new_length)],
        #             ),
        #         )

        if len(tree_nodes) >= max_nodes_to_explore:
            print(
                f"Exiting search as number of explored nodes = {len(tree_nodes)} has exceeded the limit {max_nodes_to_explore}"
            )
            break

    return False, path

# Define State class for the search tree
class State:
    def __init__(self, state, word_lengths, path_from_root = [], move = None):
        self.state = np.copy(state)
        self.word_lengths = np.copy(word_lengths)
        if move == None:
            self.path = path_from_root
        else:
            self.path = path_from_root + [(move, int(sum(word_lengths)))] # Path from root node.
        self.depth = len(self.path)

    def get_children(self, max_relator_length):
        children = []
        for action in range(0, 12):
            new_state, new_lengths, changed = ACMove(
                action,
                self.state,
                max_relator_length,
                self.word_lengths,
                cyclical=False,
            )
            
            if not changed:
                continue
            
            new_length = sum(new_lengths)
            if new_length == 2:
                return True, [], self.path + [(action, int(new_length))]

            children += [State(new_state, new_lengths, self.path, action)]
    
        return False, children, []
    
    def propagate(self, n, path_to_root, tree_nodes, max_relator_length, heap):
        state_tup, new_length = tuple(self.state), sum(self.word_lengths)
        if self.depth > 0 and state_tup not in tree_nodes:
            heapq.heappush(
            heap,
            (
                new_length,
                len(path_to_root) + n,
                state_tup,
                tuple(self.word_lengths),
                path_to_root + self.path,
            ),
            )
            tree_nodes.add(state_tup)
        if self.depth < n:
            trivial, children, path = self.get_children(max_relator_length)
            if trivial:
                return True, path_to_root + path
            else:
                for child in children:
                    trivial, path = child.propagate(n, path_to_root, tree_nodes, max_relator_length, heap)
                    if trivial:
                        return True, path
                    
        return False, []

if __name__ == "__main__":

    # presentation = np.array([1, 1, -2, -2, -2, 0, 0, 0, 0, 0, 1, 2, 1, -2, -1, -2, 0, 0, 0, 0])  # AK(2)
    presentation = np.array([-1, 2, 2, 2, 1, -2, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, 1, -2, -1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # presentation = np.array([-1, 2, 1, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, 2, 1, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    ans, path = greedy_search(presentation=presentation, max_nodes_to_explore=int(1e7), n=3)

    if path:
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
