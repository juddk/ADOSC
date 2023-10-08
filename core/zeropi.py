import scqubits as sc
import torch
from torch import sparse as ts
import numpy as np
import scipy as sp
from discretization import DOM
import xitorch
from xitorch import linalg
import utils as utl
from scipy import sparse as sps
import arbitrary as arb


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
        sparse: bool = True,
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
        self.sparse = sparse

        """
        Creation of Zero Pi Hamiltonians and Operators 

        Supports optimisation over EJ, EL

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
        # phi --> x1 and theta --> x2
        return DOM(x1=self.discretization_dim, x2=self.discretization_dim, sparse=self.sparse)

    def manual_discretization_H(self) -> torch.Tensor:
        # Constructs Hamiltonian by disretizating symbolic form given by
        # https://scqubits.readthedocs.io/en/latest/guide/qubits/zeropi.html

        # consider using https://core.ac.uk/download/pdf/144148784.pdf

        partial_phi_fd = self.init_grid().partial_x1_fd()
        partial_theta_fd = self.init_grid().partial_x2_fd()

        if self.sparse == False:
            I = torch.kron(self.init_grid().eye_x1(), self.init_grid().eye_x2())
            partial_phi_squared = torch.mm(self.init_grid().partial_x1_fd(), self.init_grid().partial_x1_bk())
            partial_theta_squared = torch.mm(self.init_grid().partial_x2_fd(), self.init_grid().partial_x2_bk())

            return (
                -2 * self.ECJ * partial_phi_squared
                # Note in line 2, we have dropped a term as it is not hermitian - 2j * self.ng * partial_theta_fd
                + 2 * self.ECS * (-1 * partial_theta_squared + self.ng**2 * I)
                + 2 * self.ECS * self.dCJ * torch.mm(partial_phi_fd, partial_theta_fd)
                - 2
                * self.EJ
                * torch.mm(self.cos_theta_operator(), self.cos_phi_operator(x=-2.0 * np.pi * self.flux / 2.0))
                + self.EL * torch.mm(self.phi_operator(), self.phi_operator())
                + 2 * self.EJ * I
                + self.EJ
                * self.dEJ
                * torch.mm(self.sin_theta_operator(), self.sin_phi_operator(x=-2.0 * np.pi * self.flux / 2.0))
            )

        else:
            I = utl.torch_sparse_kron(self.init_grid().eye_x1(), self.init_grid().eye_x2())
            partial_phi_squared = ts.mm(self.init_grid().partial_x1_fd(), self.init_grid().partial_x1_bk())
            partial_theta_squared = ts.mm(self.init_grid().partial_x2_fd(), self.init_grid().partial_x2_bk())

            return (
                -2 * self.ECJ * partial_phi_squared
                # Note in line 2, we have dropped a term as it is not hermitian - 2j * self.ng * partial_theta_fd
                + 2 * self.ECS * (-1 * partial_theta_squared + self.ng**2 * I)
                + 2 * self.ECS * self.dCJ * ts.mm(partial_phi_fd, partial_theta_fd)
                - 2
                * self.EJ
                * ts.mm(self.cos_theta_operator(), self.cos_phi_operator(x=-2.0 * np.pi * self.flux / 2.0))
                + self.EL * ts.mm(self.phi_operator(), self.phi_operator())
                + 2 * self.EJ * I
                + self.EJ
                * self.dEJ
                * ts.mm(self.sin_theta_operator(), self.sin_phi_operator(x=-2.0 * np.pi * self.flux / 2.0))
            )

    def auto_discretization_H(self) -> torch.Tensor:
        # Constructs Hamiltonian using disretization of symbolic form.
        # calls aribtrary circuit H_expression to get form of H

        zp_yaml = """# zero-pi
                branches:
                - ["JJ", 1,2, EJ1, EC1 = 20]
                - ["JJ", 3,4, EJ2=5, EC2 = 30]
                - ["L", 2,3, L1 = 0.008]
                - ["L", 4,1, L2=0.1]
                - ["C", 1,3, C1 = 0.02]
                - ["C", 2,4, C2 = 0.4]
                """
        arb.H_expression(zp_yaml)

        # calls arb discretisation function

        return

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
        if self.hamiltonian_creation_solution == "auto_H":
            eigvals, eigvecs = sp.linalg.eigh(self.auto_H())

        elif self.hamiltonian_creation_solution == "manual_discretization_davidson":
            H = (
                self.manual_discretization_H().to_dense()
                if self.manual_discretization_H().is_sparse
                else self.manual_discretization_H()
            )

            H = xitorch.LinearOperator.m(H)
            xitorch.LinearOperator._getparamnames(H, "EJ, EL")
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
    def phi_operator(self) -> torch.Tensor:
        phi = np.linspace(0, 2 * np.pi, self.discretization_dim)
        phi_m = np.diag(phi)
        if self.sparse == False:
            return torch.kron(torch.tensor(phi_m), self.init_grid().eye_x1())
        else:
            return utl.torch_sparse_kron(torch.tensor(phi_m).to_sparse(), self.init_grid().eye_x1())

    def cos_phi_operator(self, x=0.0) -> torch.Tensor:
        phi = np.linspace(0, 2 * np.pi, self.discretization_dim)
        cos_phi = np.cos(phi + x)
        cos_phi_m = np.diag(cos_phi)

        if self.sparse == False:
            return torch.kron(torch.tensor(cos_phi_m), self.init_grid().eye_x1())
        else:
            return utl.torch_sparse_kron(torch.tensor(cos_phi_m).to_sparse(), self.init_grid().eye_x1())

    def sin_phi_operator(self, x=0.0) -> torch.Tensor:
        phi = np.linspace(0, 2 * np.pi, self.discretization_dim)
        sin_phi_adj = np.sin(phi + x)
        sin_phi_adj_m = np.diag(sin_phi_adj)
        if self.sparse == False:
            return torch.kron(torch.tensor(sin_phi_adj_m), self.init_grid().eye_x1())
        else:
            return utl.torch_sparse_kron(torch.tensor(sin_phi_adj_m).to_sparse(), self.init_grid().eye_x1())

    def cos_theta_operator(self, x=0.0) -> torch.Tensor:
        theta = np.linspace(0, 2 * np.pi, self.discretization_dim)
        cos_theta_adj = np.cos(theta + x)
        cos_theta_adj_m = np.diag(cos_theta_adj)

        if self.sparse == False:
            return torch.kron(torch.tensor(cos_theta_adj_m), self.init_grid().eye_x2())
        else:
            return utl.torch_sparse_kron(torch.tensor(cos_theta_adj_m).to_sparse(), self.init_grid().eye_x2())

    def sin_theta_operator(self, x=0.0) -> torch.Tensor:
        theta = np.linspace(0, 2 * np.pi, self.discretization_dim)
        sin_theta = np.sin(theta + x)
        sin_theta_m = np.diag(sin_theta)

        if self.sparse == False:
            return torch.kron(torch.tensor(sin_theta_m), self.init_grid().eye_x1())
        else:
            return utl.torch_sparse_kron(torch.tensor(sin_theta_m).to_sparse(), self.init_grid().eye_x1())

    # Taken from analytical expression
    def d_hamiltonian_d_EJ_operator(self) -> torch.Tensor:
        if self.sparse == False:
            I = torch.kron(self.init_grid().eye_x1(), self.init_grid().eye_x2())
            return (
                -2 * torch.mm(self.cos_phi_operator(), self.cos_theta_operator(x=-2.0 * np.pi * self.flux / 2.0))
                + 2 * I
                + self.dEJ
                * torch.mm(self.sin_theta_operator(), self.sin_phi_operator(x=-2.0 * np.pi * self.flux / 2.0))
            )

        else:
            I = utl.torch_sparse_kron(self.init_grid().eye_x1(), self.init_grid().eye_x2())
            return (
                -2
                * ts.mm(
                    self.cos_phi_operator(),
                    self.cos_theta_operator(x=-2.0 * np.pi * self.flux / 2.0),
                )
                + 2 * I.to_sparse()
                + self.dEJ
                * ts.mm(
                    self.sin_theta_operator(),
                    self.sin_phi_operator(x=-2.0 * np.pi * self.flux / 2.0),
                )
            )

    def d_hamiltonian_d_flux_operator(self) -> torch.Tensor:
        if self.sparse == False:
            return self.EJ * torch.mm(
                self.cos_phi_operator(), self.sin_theta_operator(x=-2.0 * np.pi * self.flux / 2.0)
            ) - 0.5 * self.EJ * self.dEJ * torch.mm(
                self.sin_theta_operator(), self.cos_phi_operator(x=-2.0 * np.pi * self.flux / 2.0)
            )
        else:
            return self.EJ * ts.mm(
                self.cos_phi_operator(),
                self.sin_theta_operator(x=-2.0 * np.pi * self.flux / 2.0),
            ) - 0.5 * self.EJ * self.dEJ * ts.mm(
                self.sin_theta_operator(),
                self.cos_phi_operator(x=-2.0 * np.pi * self.flux / 2.0),
            )
