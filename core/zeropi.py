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
        phi_ext: torch.Tensor,
        varphi_ext: torch.Tensor,
        ng: float,
        ncut: float,
        discretization_dim: float,
        hamiltonian_creation_solution="manual_discretization",
    ):
        self.EJ = EJ
        self.EL = EL
        self.ECJ = ECJ
        self.ECS = ECS
        self.EC = EC
        self.dEJ = dEJ
        self.dCJ = dCJ
        self.phi_ext = phi_ext
        self.varphi_ext = varphi_ext
        self.ng = ng
        self.ncut = ncut
        self.discretization_dim = discretization_dim
        self.hamiltonian_creation_solution = hamiltonian_creation_solution

        """
        Creation of Zero Pi Hamiltonians and Operators 

        Parameters
        ----------
        dddd :  diddd
        ddd :   flddd

        """

    # CREATING QUBIT HAMILTONIAN
    def auto_H(self) -> np.ndarray:
        create_qubit = sc.ZeroPi(
            grid=sc.Grid1d(min_val=self.min_val, max_val=self.max_val, pt_count=self.pt_count),
            EJ=self.EJ.item(),
            EL=self.EL.item(),
            ECJ=self.ECJ.item(),
            EC=self.EC.item(),
            ng=self.ng,
            flux=self.phi_ext.item(),
            ncut=self.ncut,
            dEJ=self.dEJ.item(),
            dCJ=self.dCJ.item(),
        )
        return create_qubit.hamiltonian().toarray()

    def manual_discretization_H(self) -> torch.Tensor:
        # Constructs Hamiltonian by disretizating symbolic form given by
        # https://scqubits.readthedocs.io/en/latest/guide/qubits/zeropi.html

        I = torch.kron(torch.tensor(DOM.eye_Nphi), torch.tensor(DOM.eye_Ntheta))

        partial_phi_squared = DOM.partial_x1_fd(self.discretization_dim, self.discretization_dim) * DOM.partial_x1_bk(
            self.discretization_dim, self.discretization_dim
        )
        partial_theta_fd = DOM.partial_x2_fd(self.discretization_dim, self.discretization_dim)
        partial_phi_fd = DOM.partial_x1_fd(self.discretization_dim, self.discretization_dim)

        ham = (
            -2 * self.ECJ * partial_phi_squared
            + 2 * self.ECS * (-1 * partial_theta_fd**2 + self.ng**2 * I - 2 * self.ng * partial_theta_fd)
            + 2 * self.ECS * self.dCJ * partial_phi_fd * partial_theta_fd
            - 2 * self.EJ * self._cos_phi() * self._cos_theta_adj()
            + self.EL * self._phi() ** 2
            + 2 * self.EJ * I
            + self.EJ * self.dEJ * self._sin_theta() * self._sin_phi_adj()
        )
        return ham + torch.transpose(ham, 1, 0)

    def t1_supported_noise_channels(self):
        t1_supported_noise_channels = []
        qubit = sc.ZeroPi(
            grid=sc.Grid1d(min_val=self.min_val, max_val=self.max_val, pt_count=self.pt_count),
            EJ=self.EJ.item(),
            EL=self.EL.item(),
            ECJ=self.ECJ.item(),
            EC=self.EC.item(),
            ng=self.ng,
            flux=self.phi_ext.item(),
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
            grid=sc.Grid1d(min_val=self.min_val, max_val=self.max_val, pt_count=self.pt_count),
            EJ=self.EJ.item(),
            EL=self.EL.item(),
            ECJ=self.ECJ.item(),
            EC=self.EC.item(),
            ng=self.ng,
            flux=self.phi_ext.item(),
            ncut=self.ncut,
            dEJ=self.dEJ,
            dCJ=self.dCJ,
        )
        for x in qubit.supported_noise_channels():
            if x.startswith("tphi"):
                tphi_supported_noise_channels.append(x)
        return tphi_supported_noise_channels

    def esys(self):
        # add a kwargs to chnage variables in xitorch
        if self.hamiltonian_creation_solution == "auto_H":
            eigvals, eigvecs = sp.linalg.eigh(self.auto_H())
        elif self.hamiltonian_creation_solution == "manual_discretization":
            H = xitorch.LinearOperator.m(self.manual_discretization_H())
            xitorch.LinearOperator._getparamnames(H, "EJ, EJ, EC")
            eigvals, eigvecs = xitorch.linalg.symeig(
                H,
                2,
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
    def _phi(self) -> torch.Tensor:
        phi = np.linspace(0, 2 * np.pi, DOM.Nphi)
        phi_m = np.diag(phi)
        return torch.kron(torch.tensor(phi_m), torch.tensor(DOM.eye_Nphi))

    def _cos_phi(self) -> torch.Tensor:
        phi = np.linspace(0, 2 * np.pi, DOM.Nphi)
        cos_phi = np.cos(phi)
        cos_phi_m = np.diag(cos_phi)
        return torch.kron(torch.tensor(cos_phi_m), torch.tensor(DOM.eye_Nphi))

    def _sin_phi_adj(self) -> torch.Tensor:
        phi = np.linspace(0, 2 * np.pi, DOM.Nphi)
        sin_phi_adj = np.sin(phi - self.phi_ext.detach().numpy() / 2)
        sin_phi_adj_m = np.diag(sin_phi_adj)
        return torch.kron(torch.tensor(sin_phi_adj_m), torch.tensor(DOM.eye_Nphi))

    def _cos_theta_adj(self) -> torch.Tensor:
        theta = np.linspace(0, 2 * np.pi, DOM.Ntheta)
        cos_theta_adj = np.cos(theta - self.varphi_ext.detach().numpy() / 2)
        cos_theta_adj_m = np.diag(cos_theta_adj)
        return torch.kron(torch.tensor(cos_theta_adj_m), torch.tensor(DOM.eye_Ntheta))

    def _sin_theta(self) -> torch.Tensor:
        theta = np.linspace(0, 2 * np.pi, DOM.Ntheta)
        sin_theta = np.sin(theta)
        sin_theta_m = np.diag(sin_theta)
        return torch.kron(torch.tensor(sin_theta_m), torch.tensor(DOM.eye_Ntheta))

    # WIP --> DO THIS IN SCALABLE WAY BY DIFFERENTATING THE SCQUBITS HAMILTONIAN OUTPUT

    def d_hamiltonian_d_EJ(self) -> torch.Tensor:
        d_potential_d_EJ_mat = -2.0 * torch.kron(
            self._cos_phi_operator(x=-2.0 * np.pi * self.phi_ext.item() / 2.0),
            self._cos_theta_operator(),
        )

        ###the flux.item() could be an issue for computing gradients
        return d_potential_d_EJ_mat

    def d_hamiltonian_d_flux(self) -> torch.Tensor:
        op_1 = torch.kron(
            self._sin_phi_operator(x=-2.0 * np.pi * self.phi_ext.item() / 2.0),
            self._cos_theta_operator(),
        )
        op_2 = torch.kron(
            self._cos_phi_operator(x=-2.0 * np.pi * self.phi_ext.item() / 2.0),
            self._sin_theta_operator(),
        )
        d_potential_d_flux_mat = -2.0 * np.pi * self.EJ * op_1 - np.pi * self.EJ * self.dEJ * op_2

        ###the flux.item() could be an issue for computing gradients
        return d_potential_d_flux_mat
