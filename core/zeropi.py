import scqubits as sc
import torch
import numpy as np
from scipy import sparse
from scipy.sparse import dia_matrix
import general
import scipy as sp
from typing import Union
from discretization import DOM
import xitorch
from xitorch import linalg
import utils as utl


class ZeroPi:
    # All values in GHz
    def __init__(
        self,
        EJ: torch.Tensor,
        EL: torch.Tensor,
        ECJ: torch.Tensor,
        ECS: torch.Tensor,
        EC: torch.Tensor,
        dEJ: torch.Tensor,
        dCJ: torch.Tensor,
        flux: float,
        ng: float,
        ncut: float,
        discretization_dim: float,
        hamiltonian_creation_solution: str = "manual_discretization_davidson",
    ):
        self.EJ = EJ
        self.EL = EL
        self.ECJ = ECJ
        self.ECS = ECS
        self.EC = EC
        self.dEJ = dEJ
        self.dCJ = dCJ
        self.flux = flux
        self.ng = ng
        self.ncut = ncut
        self.discretization_dim = discretization_dim
        self.hamiltonian_creation_solution = hamiltonian_creation_solution

        """
        Creation of Zero Pi Hamiltonians and Operators 

        Supports optimisation over EJ, EL, ECJ, ECS, EC, dEJ, dCJ. 

        Parameters
        ----------

        dddd :  diddd
        ddd :   flddd

        """

    # CREATING QUBIT HAMILTONIAN
    def auto_H(self) -> np.ndarray:
        create_qubit = sc.ZeroPi(
            grid=sc.Grid1d(min_val=-np.pi / 3, max_val=np.pi / 2, pt_count=self.discretization_dim),
            EJ=self.EJ.item(),
            EL=self.EL.item(),
            ECJ=self.ECJ.item(),
            EC=self.EC.item(),
            ng=self.ng,
            flux=self.flux,
            ncut=self.ncut,
            dEJ=self.dEJ.item(),
            dCJ=self.dCJ.item(),
        )
        return create_qubit.hamiltonian().toarray()

    def init_grid(self):
        # Initialise discretization grid
        # phi maps to x1 and theta maps to x2
        return DOM(x1=self.discretization_dim, x2=self.discretization_dim)

    def manual_discretization_H(self, sparse: bool = True) -> torch.Tensor:
        # Constructs Hamiltonian by disretizating symbolic form given by
        # https://scqubits.readthedocs.io/en/latest/guide/qubits/zeropi.html

        I = torch.kron(self.init_grid().eye_x1(), self.init_grid().eye_x2())
        partial_phi_squared = self.init_grid().partial_x1_fd() * self.init_grid().partial_x1_bk()
        partial_theta_fd = self.init_grid().partial_x2_fd()
        partial_phi_fd = self.init_grid().partial_x1_fd()

        ham = (
            -2 * self.ECJ * partial_phi_squared
            + 2 * self.ECS * (-1 * partial_theta_fd**2 + self.ng**2 * I - 2 * self.ng * partial_theta_fd)
            + 2 * self.ECS * self.dCJ * partial_phi_fd * partial_theta_fd
            - 2 * self.EJ * self.cos_phi_operator() * self.cos_theta_operator(x=-2.0 * np.pi * self.flux / 2.0)
            + self.EL * self.phi_operator() ** 2
            + 2 * self.EJ * I
            + self.EJ * self.dEJ * self.sin_theta_operator() * self.sin_phi_operator(x=-2.0 * np.pi * self.flux / 2.0)
        )
        H = ham + torch.transpose(ham, 1, 0)
        if sparse == False:
            return H
        if sparse == True:
            return H.to_sparse()

    def t1_supported_noise_channels(self):
        t1_supported_noise_channels = []
        qubit = sc.ZeroPi(
            grid=sc.Grid1d(min_val=-np.pi / 3, max_val=np.pi / 2, pt_count=self.discretization_dim),
            EJ=self.EJ.item(),
            EL=self.EL.item(),
            ECJ=self.ECJ.item(),
            EC=self.EC.item(),
            ng=self.ng,
            flux=self.flux,
            ncut=self.ncut,
            dEJ=self.dEJ,
            dCJ=self.dCJ,
        )
        for x in qubit.supported_noise_channels():
            if x.startswith("t1"):
                t1_supported_noise_channels.append(x)
        return t1_supported_noise_channels

    def tphi_supported_noise_channels(self):
        tphi_supported_noise_channels = []
        qubit = sc.ZeroPi(
            grid=sc.Grid1d(min_val=-np.pi / 3, max_val=np.pi / 2, pt_count=self.discretization_dim),
            EJ=self.EJ.item(),
            EL=self.EL.item(),
            ECJ=self.ECJ.item(),
            EC=self.EC.item(),
            ng=self.ng,
            flux=self.flux,
            ncut=self.ncut,
            dEJ=self.dEJ,
            dCJ=self.dCJ,
        )
        for x in qubit.supported_noise_channels():
            if x.startswith("tphi"):
                tphi_supported_noise_channels.append(x)
        return tphi_supported_noise_channels

    def esys(self):
        # add a kwargs to chnage variables in xitorch,
        # should be able to choose number of eigenvalues,
        # think currently selects the largest but we would want the smallest

        if self.hamiltonian_creation_solution == "auto_H":
            eigvals, eigvecs = sp.linalg.eigh(self.auto_H(sparse=False))

        elif self.hamiltonian_creation_solution == "manual_discretization_davidson":
            H = xitorch.LinearOperator.m(self.manual_discretization_H(sparse=False))
            xitorch.LinearOperator._getparamnames(H, "EJ, EL, ECJ, ECS, EC, dEJ, dCJ")
            eigvals, eigvecs = xitorch.linalg.symeig(
                A=H,
                neig=2,
                mode="lowest",
                method="davidson",
                max_niter=100,
                nguess=None,
                v_init="randn",
                max_addition=None,
                min_eps=1e-06,
                verbose=False,
            )
        return eigvals, eigvecs

    # Operators
    def phi_operator(self, sparse: bool = True) -> torch.Tensor:
        phi = np.linspace(0, 2 * np.pi, self.discretization_dim)
        phi_m = np.diag(phi)
        O = torch.kron(torch.tensor(phi_m), torch.tensor(self.init_grid().eye_x1()))
        if sparse == False:
            return O
        if sparse == True:
            return O.to_sparse()

    def cos_phi_operator(self, x=0.0, sparse: bool = True) -> torch.Tensor:
        phi = np.linspace(0, 2 * np.pi, self.discretization_dim)
        cos_phi = np.cos(phi + x)
        cos_phi_m = np.diag(cos_phi)
        O = torch.kron(torch.tensor(cos_phi_m), torch.tensor(self.init_grid().eye_x1()))
        if sparse == False:
            return O
        if sparse == True:
            return O.to_sparse()

    def sin_phi_operator(self, x=0.0, sparse: bool = True) -> torch.Tensor:
        phi = np.linspace(0, 2 * np.pi, self.discretization_dim)
        sin_phi_adj = np.sin(phi + x)
        sin_phi_adj_m = np.diag(sin_phi_adj)
        O = torch.kron(torch.tensor(sin_phi_adj_m), torch.tensor(self.init_grid().eye_x1()))
        if sparse == False:
            return O
        if sparse == True:
            return O.to_sparse()

    # this is different to scqubits - potentially others are too?
    def cos_theta_operator(self, x=0.0, sparse: bool = True) -> torch.Tensor:
        theta = np.linspace(0, 2 * np.pi, self.discretization_dim)
        cos_theta_adj = np.cos(theta + x)
        cos_theta_adj_m = np.diag(cos_theta_adj)
        O = torch.kron(torch.tensor(cos_theta_adj_m), torch.tensor(self.init_grid().eye_x2()))
        if sparse == False:
            return O
        if sparse == True:
            return O.to_sparse()

    def sin_theta_operator(self, x=0.0, sparse: bool = True) -> torch.Tensor:
        theta = np.linspace(0, 2 * np.pi, self.discretization_dim)
        sin_theta = np.sin(theta + x)
        sin_theta_m = np.diag(sin_theta)
        O = torch.kron(torch.tensor(sin_theta_m), torch.tensor(self.init_grid().eye_x1()))
        if sparse == False:
            return O
        if sparse == True:
            return O.to_sparse()

    # Taken from analytical expression
    def d_hamiltonian_d_EJ_operator(self, sparse: bool = True) -> torch.Tensor:
        I = torch.kron(self.init_grid().eye_x1(), self.init_grid().eye_x2())

        if sparse == False:
            return (
                -2
                * self.cos_phi_operator(sparse=sparse)
                * self.cos_theta_operator(x=-2.0 * np.pi * self.flux / 2.0, sparse=sparse)
                + 2 * I
                + self.dEJ
                * self.sin_theta_operator(sparse=sparse)
                * self.sin_phi_operator(x=-2.0 * np.pi * self.flux / 2.0, sparse=sparse)
            )

        else:
            return (
                -2
                * utl.sparse_mv(
                    self.cos_phi_operator(sparse=sparse),
                    self.cos_theta_operator(x=-2.0 * np.pi * self.flux / 2.0, sparse=sparse),
                )
                + 2 * I.to_sparse()
                + self.dEJ
                * utl.sparse_mv(
                    self.sin_theta_operator(sparse=sparse),
                    self.sin_phi_operator(x=-2.0 * np.pi * self.flux / 2.0, sparse=sparse),
                )
            )

    def d_hamiltonian_d_flux_operator(self, sparse: bool = True) -> torch.Tensor:
        if sparse == False:
            return self.EJ * self.cos_phi_operator(sparse=sparse) * self.sin_theta_operator(
                x=-2.0 * np.pi * self.flux / 2.0, sparse=sparse
            ) - 0.5 * self.EJ * self.dEJ * self.sin_theta_operator(sparse=sparse) * self.cos_phi_operator(
                x=-2.0 * np.pi * self.flux / 2.0, sparse=sparse
            )
        if sparse == True:
            return self.EJ * utl.sparse_mv(
                self.cos_phi_operator(sparse=sparse),
                self.sin_theta_operator(x=-2.0 * np.pi * self.flux / 2.0, sparse=sparse),
            ) - 0.5 * self.EJ * self.dEJ * utl.sparse_mv(
                self.sin_theta_operator(sparse=sparse),
                self.cos_phi_operator(x=-2.0 * np.pi * self.flux / 2.0, sparse=sparse),
            )

    """ 
    def d_hamiltonian_d_EJ(self) -> torch.Tensor:
        d_hamiltonian_d_EJ = -2.0 * torch.kron(
            self._cos_phi(x=-2.0 * np.pi * self.flux / 2.0),
            self._cos_theta(),
        )
        return d_hamiltonian_d_EJ

    def d_hamiltonian_d_flux(self) -> torch.Tensor:
        op_1 = torch.kron(
            self._sin_phi(x=-2.0 * np.pi * self.flux / 2.0),
            self._cos_theta(),
        )
        op_2 = torch.kron(
            self._cos_phi(x=-2.0 * np.pi * self.flux / 2.0),
            self._sin_theta(),
        )
        return -2.0 * np.pi * self.EJ * op_1 - np.pi * self.EJ * self.dEJ * op_2
"""
