from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.

    vals_forward = list(vals)
    vals_backward = list(vals)
    vals_forward[arg] += epsilon
    vals_backward[arg] -= epsilon

    f_forward = f(*vals_forward)
    f_backward = f(*vals_backward)

    derivative = (f_forward - f_backward) / (2 * epsilon)
    
    return derivative


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    visited = set()
    sorted_nodes = []

    def dfs(curr: Variable) -> None:
        if curr.unique_id in visited or curr.is_constant():
            return
        visited.add(curr.unique_id)

        for neigh in curr.parents:
            dfs(neigh)

        sorted_nodes.append(curr)

    dfs(variable)
    return reversed(sorted_nodes)

         
        


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    sorted_nodes = topological_sort(variable)
    gradients = {variable.unique_id: deriv}

    for v in sorted_nodes:
        d_output = gradients.get(v.unique_id, 0)

        if v.is_leaf():
            v.accumulate_derivative(d_output)
        else:
            for parent, local_d in v.chain_rule(d_output):
                if parent.unique_id not in gradients.keys():
                    gradients[parent.unique_id] = 0
                gradients[parent.unique_id] += local_d

                


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
