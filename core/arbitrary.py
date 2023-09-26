import scqubit
import sympy as spy
from sympy import symbols, lambdify, Expr
import torch


class AribtraryCircuit:
    def __init__(self, circuit_yaml):
        self.circuit_yaml = circuit_yaml

    """
    Creation of Aribtrary Circuit Hamiltonians and Operators 

    Parameters
    ----------
    circuit_yaml : yaml file defining custom circuit
    
    e.g 

    zp_yaml =
    - ["JJ", 1,2, EJ = 10, 20]
    - ["JJ", 3,4, EJ, 20]
    - ["L", 2,3, 0.008]
    - ["L", 4,1, 0.008]
    - ["C", 1,3, 0.02]
    - ["C", 2,4, 0.02]
    
    comment on when it works re the paramater values 

    """

    def H_expression() -> spy.Expr:
        # just calls scqubits
        return

    def dH_expression() -> spy.Expr:
        # function to find dH/dEj or dH/dflux etc for use in coherence time calc
        return

    def discretize_operator() -> torch.Tensor:
        # function to discretize any operator e.g a form_of_H or
        return
