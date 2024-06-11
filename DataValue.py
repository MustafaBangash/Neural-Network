from math import exp

class DataValue:

    def __init__(self, data, _children=(), _operator=''):
        self.data = data
        self.gradient = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._operator = _operator # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, DataValue) else DataValue(other)
        out = DataValue(self.data + other.data, (self, other), '+')

        def _backward():
            self.gradient += out.gradient
            other.gradient += out.gradient
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, DataValue) else DataValue(other)
        out = DataValue(self.data * other.data, (self, other), '*')

        def _backward():
            self.gradient += other.data * out.gradient
            other.gradient += self.data * out.gradient
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = DataValue(self.data**other, (self,), f'**{other}')

        def _backward():
            self.gradient += (other * self.data**(other-1)) * out.gradient
        out._backward = _backward

        return out

    def relu(self):
        out = DataValue(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.gradient += (out.data > 0) * out.gradient
        out._backward = _backward

        return out

    def sigmoid(self):
        sigmoid = 1/(1 + exp(-self.data))
        out = DataValue(sigmoid, (self,), 'Sigmoid')

        def _backward():
            self.gradient += sigmoid * (1 - sigmoid) * out.gradient

        out._backward = _backward

        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradientient
        self.gradient = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, gradient={self.gradient})"