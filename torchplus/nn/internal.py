"""
Where the TorchPlus layer magic lives.
"""

from collections import defaultdict
from torch.nn.modules import Module as PyTorchModule
from torch.nn import Sequential as PyTorchSequential


class PlusCache(object):
    """
    The cache that is used to implement chaining nodes with the + operator.

    It works by assigning and propagating forward a sequence ID for each node in
    the "+" chains, and packaging the observed node chains into sequence nodes
    that are returned to the caller.

    Behind the scenes, we create a global singleton of this class, which is
    called from the node base class (PseudoNode)'s __add__ operator.  So don't
    use "+" syntactic sugar from different threads at the same time.  This could
    probably be worked around if it were ever an issue.
    """

    def __init__(self):
        """
        Set up the cache with no contents.
        """
        self.next_seq_id = 1
        self.node2seqid = {}
        self.seqid2nodes = defaultdict(list)
        self.prev_right = None

    def connect(self, left, right):
        """
        Evaluate one "left + right" operation, returning a new sequence node.

        The Returned node will be immediately thrown away unless this is the
        last comparison of the "+" chain.

        This is only called from PseudoNode.__add__.
        """
        # Save the new previous right node.
        self.prev_right = right

        # Either retrieve or invent the sequence ID of the node on the left.
        seq_id = self.node2seqid.get(left)
        if seq_id is None:
            seq_id = self.next_seq_id
            self.next_seq_id += 1
            self.node2seqid[left] = seq_id
            self.seqid2nodes[seq_id].append(left)

        # Propagate the left node's sequence ID forward to the right node.
        self.node2seqid[right] = seq_id
        self.seqid2nodes[seq_id].append(right)

        # Return a Sequential of the nodes of that sequence ID.
        nodes = self.seqid2nodes[seq_id]
        nodes = map(lambda node: node.instance().convert(), nodes)
        return Node(PyTorchSequential(*nodes))


PLUS_CACHE = PlusCache()


class PseudoNode(PyTorchModule):
    """
    A node-like building block for constructing neural networks.

    This is the base class of node types, which wrap "real" PyTorch nodes.
    PseudoNode exists because of Keyword, otherwise there would just be Nodes.
    """

    def __add__(self, right):
        """
        Accumulate this node and its neighbor into a node sequence.
        """
        return PLUS_CACHE.connect(self, right)

    def instance(self):
        """
        Desugar fake node class names.
        """
        raise NotImplementedError


class Node(PseudoNode):
    """
    A magic wrapper around a PyTorch layer that enables "+" sequences.
    """

    def __init__(self, wrapped):
        """
        Keep the real PyTorch module inside.
        """
        PseudoNode.__init__(self)
        self.wrapped = wrapped
        self.__dict__['wrapped_ref'] = wrapped

    def instance(self):
        """
        We are an instance already.
        """
        return self

    def convert(self):
        """
        Dynamically turn into the PyTorch layer we contain inside for use.
        """
        wrapped = self.__dict__['wrapped_ref']
        self.__class__ = wrapped.__class__
        self.__dict__ = wrapped.__dict__
        return self

    def __call__(self, *args, **kwargs):
        """
        Turn into our contained layer, then __call__().
        """
        self.convert()
        return self.__call__(*args, **kwargs)

    def __dir__(self, *args, **kwargs):
        """
        Turn into our contained layer, then __dir__().
        """
        self.convert()
        return self.__dir__(*args, **kwargs)

    def __repr__(self, *args, **kwargs):
        """
        Turn into our contained layer, then __repr__().
        """
        self.convert()
        return self.__repr__(*args, **kwargs)

    def __getattr__(self, key):
        """
        Turn into our contained layer, then getattr().
        """
        self.convert()
        return getattr(self, key)


class Keyword(PseudoNode):
    """
    A magic type that enables "+" sequences and creating layers without parens.

    Instances of Keyword are basically fake PyTorch layer class names,
    """

    def __init__(self, klass):
        """
        Keep the PyTorch layer class we are a factory for inside.
        """
        PseudoNode.__init__(self)
        self.klass = klass

    def __call__(self, *args, **kwargs):
        """
        Fake constructor: create the PyTorch layer and wrap it in a new Node.
        """
        module = self.klass(*args, **kwargs)
        return Node(module)

    def instance(self):
        """
        Get an instance of the actual layer this builds.
        """
        return self.__call__()
