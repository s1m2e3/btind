"""Tree node for the recursive-growth experiments (e02, e13).

RECONSTRUCTED. The original of this file was destroyed by an overwrite and
there was no version control to recover it from. What follows is rebuilt from
its three call sites -- `joint.grow_joint`, `joint.grow_veto` and
`valuesplit.grow_value` -- which construct a node with (n, depth, theta, guard),
set (split_var, split_at, stat, left, right) when they split it, and read
`leaves()` to decide when to stop. It is faithful to that interface and to
nothing else; any convenience the original carried beyond it is gone.

None of those three functions is reachable from the current pipeline, which
grows its trees by rollout (`grow_bt.py`) rather than by recursive partitioning.
They are kept because the experiments that produced the project's early results
called them.
"""


class Node:
    """One node of a binary partition: a region, its law, and how it was cut."""

    def __init__(self, n=0, depth=0, theta=None, guard=None):
        self.n = n                  # rows falling in this region
        self.depth = depth
        self.theta = theta          # affine law fitted on those rows
        self.guard = list(guard or [])   # (var, threshold, is_left) path here
        self.split_var = None
        self.split_at = None
        self.stat = None            # the criterion value that bought the split
        self.left = None
        self.right = None

    def is_leaf(self):
        return self.left is None and self.right is None

    def leaves(self):
        if self.is_leaf():
            return [self]
        return self.left.leaves() + self.right.leaves()

    def __repr__(self):
        if self.is_leaf():
            return "Leaf(n=%d, depth=%d)" % (self.n, self.depth)
        return "Node(n=%d, split=%s<%.4f)" % (self.n, self.split_var,
                                              self.split_at)
