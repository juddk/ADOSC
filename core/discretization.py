import torch
import scipy as sp
from scipy import sparse as sps
import utils as utl


class DOM:
    def __init__(self, x1: int, x2: int, sparse: bool):
        self.x1 = x1
        self.x2 = x2
        self.sparse = sparse

        """
        Differential Operator Matrices

        Parameters
        ----------
        x1, x2: discretization dimension for each variable

        sparse: data format

        """

    def eye_x1(self) -> torch.Tensor:
        if self.sparse:
            return torch.eye(self.x1).to_sparse()
        else:
            return torch.eye(self.x1)

    def eye_x2(self) -> torch.Tensor:
        if self.sparse:
            return torch.eye(self.x2).to_sparse()
        else:
            return torch.eye(self.x2)

    def partial_x1_fd(self) -> torch.Tensor:
        if self.sparse:
            return utl.sps_to_torch_sparse(
                sps.kron(
                    utl.torch_sparse_to_sps(self.eye_x2()),
                    sps.diags([-1, 1, 1], [0, 1, -self.x1 + 1], shape=(self.x1, self.x1)),
                )
            )
        else:
            return torch.kron(
                self.eye_x2(),
                torch.tensor(sps.diags([-1, 1, 1], [0, 1, -self.x1 + 1], shape=(self.x1, self.x1)).todense()),
            )

    def partial_x1_bk(self) -> torch.Tensor:
        if self.sparse:
            return utl.sps_to_torch_sparse(
                sps.kron(
                    utl.torch_sparse_to_sps(self.eye_x2()),
                    sps.diags([1, -1, -1], [0, -1, self.x1 - 1], shape=(self.x1, self.x1)),
                )
            )

        else:
            return torch.kron(
                self.eye_x2(),
                torch.tensor(sps.diags([1, -1, -1], [0, -1, self.x1 - 1], shape=(self.x1, self.x1)).todense()),
            )

    def partial_x2_fd(self) -> torch.Tensor:
        if self.sparse:
            return utl.sps_to_torch_sparse(
                sps.kron(
                    utl.torch_sparse_to_sps(self.eye_x1()),
                    sps.diags([-1, 1, 1], [0, 1, -self.x2 + 1], shape=(self.x2, self.x2)),
                )
            )

        else:
            return torch.kron(
                self.eye_x1(),
                torch.tensor(sps.diags([-1, 1, 1], [0, 1, -self.x2 + 1], shape=(self.x2, self.x2)).todense()),
            )

    def partial_x2_bk(self) -> torch.Tensor:
        if self.sparse:
            return utl.sps_to_torch_sparse(
                sps.kron(
                    utl.torch_sparse_to_sps(self.eye_x1()),
                    sps.diags([1, -1, -1], [0, -1, self.x2 - 1], shape=(self.x2, self.x2)),
                )
            )

        else:
            return torch.kron(
                self.eye_x1(),
                torch.tensor(sps.diags([1, -1, -1], [0, -1, self.x2 - 1], shape=(self.x2, self.x2)).todense()),
            )
