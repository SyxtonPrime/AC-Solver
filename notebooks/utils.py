# A collection of helper functions which can be used across notebooks.

import numpy as np
import copy
 
# The mapping from the {+/- 2, +/- 1, 0} representation into a one-hot
# representation on a two dimensional space.
one_hot_single = {-2: [0, -1], -1: [-1, 0], 0: [0, 0], 1: [1, 0], 2: [0, 1]}

# Convert from the standard presentation to a one_hot encoding mapping a generator to 1 and its inverse to -1.
# We pad the output to length outlen.
def to_one_hot(presentation: np.typing.NDArray[np.int32], out_len: int) -> np.typing.NDArray[np.int32]:
    relator_len = len(presentation) // 2
    first_relator = [one_hot_single[presentation[x]] for x in range(0, relator_len)] + [[0, 0]] * (out_len - relator_len)
    second_relator = [one_hot_single[presentation[x + relator_len]] for x in range(0, relator_len)] + [[0, 0]] * (out_len - relator_len)

    relator_pair = np.array(first_relator + second_relator, dtype=np.float32)
    return relator_pair.flatten()



# Swap the two relations of a given presentation.
def swap_relations(presentation, relation_length):
    new_presentation = np.zeros(2 * relation_length)
    new_presentation[:relation_length] = presentation[relation_length:]
    new_presentation[relation_length:] = presentation[:relation_length]
    return new_presentation

# Swap the two generators of a given presentation.
def swap_generators(presentation, relation_length):
    new_presentation = np.zeros(2 * relation_length)
    for (i, elem) in enumerate(presentation):
        if elem > 0:
            # Map 2 -> 1 and 1 -> 2
            new_presentation[i] = 3 - elem
        elif elem < 0:
            # Map -2 -> -1 and -1 -> -2
            new_presentation[i] = -3 - elem
    return new_presentation

# Swap the first generator with its inverse
def invert_x(presentation, relation_length):
    new_presentation = copy.copy(presentation)
    for (i, elem) in enumerate(presentation):
        if elem == 1:
            # Map 1 -> -1
            new_presentation[i] = -1
        elif elem == -1:
            # Map -1 -> 1
            new_presentation[i] = 1

    return new_presentation

# Find all sixteen equivalent presentations related to the original by our equivariance group.
# 
# Each element of the equivariance group can be written uniquely as (g1)^{i1}*(g2)^{i2}*(g3)^{i3}*(g4)^{i4}
# where i1, i2, i3, i4 as 0 or 1 and
# g4: Swap the relations.
# g3: Swap the generators.
# g2: Swap the first generator with its inverse.
# g1: Swap both generators with their inverse.
# Note that  g1(P) = g3 * g2 * g3 * g2(P) so it isn't a "true" generator but its good enough for here.
def group_equivalency_class(presentation, relation_length):
    equivalence_class = [presentation]
    equivalence_class += [swap_relations(presentation, relation_length)]
    equivalence_class += [swap_generators(pres, relation_length) for pres in equivalence_class]
    equivalence_class += [invert_x(pres, relation_length) for pres in equivalence_class]
    equivalence_class += [-pres for pres in equivalence_class]

    return equivalence_class