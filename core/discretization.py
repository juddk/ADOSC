import torch
import scipy as sp
from scipy import sparse as sps

# dont quite understand, are x1 and x2 conjugate variables
# why do we need partial matricies for both then?
# how do we generalise this higher - is this just one set of 2 conjugate variables
# comment on the boundary conditions and why they were chosen


class DOM:
    def __init__(
        self,
        x1,
        x2,
    ):
        self.x1 = x1
        self.x2 = x2

        """
        Differential Operator Matrices

        Parameters
        ----------
        dfff  :   discretization dimension for each

        """

    def eye_x1(self) -> torch.Tensor:
        return torch.eye(self.x1)

    def eye_x2(self) -> torch.Tensor:
        return torch.eye(self.x2)

    def partial_x1_fd(self) -> torch.Tensor:
        return torch.kron(
            self.eye_x2(),
            torch.tensor(sps.diags([-1, 1, 1], [0, 1, -self.x1 + 1], shape=(self.x1, self.x1)).todense()),
        )

    def partial_x1_bk(self) -> torch.Tensor:
        return torch.kron(
            self.eye_x2(),
            torch.tensor(sps.diags([1, -1, -1], [0, -1, self.x1 - 1], shape=(self.x1, self.x1)).todense()),
        )

    def partial_x2_fd(self) -> torch.Tensor:
        return torch.kron(
            self.eye_x1(),
            torch.tensor(sps.diags([-1, 1, 1], [0, 1, -self.x2 + 1], shape=(self.x2, self.x2)).todense()),
        )

    def partial_x2_bk(self) -> torch.Tensor:
        return torch.kron(
            self.eye_x1(),
            torch.tensor(sps.diags([1, -1, -1], [0, -1, self.x2 - 1], shape=(self.x2, self.x2)).todense()),
        )
