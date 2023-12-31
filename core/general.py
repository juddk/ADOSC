import torch
import numpy as np
import scipy as sp
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast
import scipy.constants
import torch_sparse as ts
import utils as utl

# NOISE_PARAMS TAKEN FROM SCQUBITS
NOISE_PARAMS = {
    "A_flux": 1e-6,  # Flux noise strength. Units: Phi_0
    "A_ng": 1e-4,  # Charge noise strength. Units of charge e
    "A_cc": 1e-7,  # Critical current noise strength. Units of critical current I_c
    "omega_low": 1e-9 * 2 * np.pi,  # Low frequency cutoff. Units: 2pi GHz
    "omega_high": 3 * 2 * np.pi,  # High frequency cutoff. Units: 2pi GHz
    "Delta": 3.4e-4,  # Superconducting gap for aluminum (at T=0). Units: eV
    "x_qp": 3e-6,  # Quasiparticles density (see for example Pol et al 2014)
    "t_exp": 1e4,  # Measurement time. Units: ns
    "R_0": 50,  # Characteristic impedance of a transmission line. Units: Ohms
    "T": 0.015,  # Typical temperature for a superconducting circuit experiment. Units: K
    "M": 400,  # Mutual inductance between qubit and a flux line. Units: \Phi_0 / Ampere
    "R_k": sp.constants.h / (sp.constants.e**2.0)  # Normal quantum resistance, aka Klitzing constant.
    # Note, in some papers a superconducting quantum
    # resistance is used, and defined as: h/(2e)^2
}


# USEFUL FUNCTIONS


def calc_therm_ratio(omega: torch.Tensor, T: float = NOISE_PARAMS["T"]):
    # omega must be in units of radians/s
    return (sp.constants.hbar * omega * 1e9) / (sp.constants.k * T)


# USEFUL OPERATORS
def annihilation(dimension: int) -> np.ndarray:
    offdiag_elements = np.sqrt(range(1, dimension))
    return np.diagflat(offdiag_elements, 1)


def creation(dimension: int) -> np.ndarray:
    return annihilation(dimension).T


def omega(eigvals) -> torch.Tensor:
    # angular frequency between ground and excited states

    ground_E = eigvals[0]
    excited_E = eigvals[1]
    return 2 * np.pi * (excited_E - ground_E)


# GENERIC T1 AND TPHI FORMULAS
def t1_rate(
    noise_op: torch.Tensor,
    spectral_density: torch.Tensor,
    eigvecs: torch.Tensor,
) -> torch.Tensor:
    s = spectral_density
    # We have defined spectral densitites to be spectral_density(omega)+spectral_density(-omega)

    if noise_op.is_sparse:
        ground = eigvecs[:, 0].unsqueeze(1).to_sparse()
        excited = eigvecs[:, 1].unsqueeze(1).to_sparse()

        rate = utl.sparse_mv(noise_op.to(torch.complex128), ground.to(torch.complex128))

        rate = utl.sparse_mv(torch.transpose(excited.conj().to(torch.complex128), -1, 0), rate)

    else:
        ground = eigvecs[:, 0]
        excited = eigvecs[:, 1]
        rate = torch.matmul(noise_op.to(torch.complex128), torch.transpose(ground.to(torch.complex128), -1, 0))
        rate = torch.matmul(excited.conj().to(torch.complex128), rate)

    rate = torch.pow(torch.abs(rate), 2) * s

    return rate


def tphi_rate(
    A_noise: float,
    noise_op: torch.Tensor,
    eigvecs: torch.Tensor,
    omega_low: float = NOISE_PARAMS["omega_low"],
    t_exp: float = NOISE_PARAMS["t_exp"],
) -> torch.Tensor:
    if noise_op.is_sparse:
        ground = eigvecs[:, 0].unsqueeze(1).to_sparse()
        excited = eigvecs[:, 1].unsqueeze(1).to_sparse()

        right_braket = utl.sparse_mv(noise_op.to(torch.complex128), ground.to(torch.complex128))
        full_braket_1 = utl.sparse_mv(torch.transpose(ground.conj().to(torch.complex128), -1, 0), right_braket)
        full_braket_2 = utl.sparse_mv(torch.transpose(excited.conj().to(torch.complex128), -1, 0), right_braket)

        rate = abs(full_braket_1 - full_braket_2)

    else:
        ground = eigvecs[:, 0]
        excited = eigvecs[:, 1]
        rate = torch.abs(
            torch.matmul(
                ground.conj().to(torch.complex128),
                torch.matmul(noise_op.to(torch.complex128), torch.transpose(ground.to(torch.complex128), -1, 0)),
            )
            - torch.matmul(
                excited.conj().to(torch.complex128),
                torch.matmul(noise_op.to(torch.complex128), torch.transpose(ground.to(torch.complex128), -1, 0)),
            )
        )

    rate *= A_noise * np.sqrt(2 * np.abs(np.log(omega_low * t_exp)))

    # We assume that the system energies are given in units of frequency and
    # not the angular frequency, hence we have to multiply by `2\pi`
    rate *= 2 * np.pi

    return rate


# T1 AND TPHI ACROSS SPECIFIC NOISE CHANNELS
def effective_t1_rate(
    qubit,
    eigvecs: torch.Tensor,
    eigvals: torch.Tensor,
    noise_channels: Union[str, List[str]],
    T: float = NOISE_PARAMS["T"],
    M: float = NOISE_PARAMS["M"],
    R_0: float = NOISE_PARAMS["R_0"],
    R_k: float = NOISE_PARAMS["R_k"],
) -> torch.Tensor:
    t1 = torch.zeros([1, 1], dtype=torch.double)

    if "t1_capacitive" in noise_channels:
        t1 += t1_rate(
            noise_op=qubit.n_operator(),
            spectral_density=spectral_density_cap(qubit, eigvals, True, T)
            + spectral_density_cap(qubit, eigvals, False, T),
            eigvecs=eigvecs,
        )

    if "t1_flux_bias_line" in noise_channels:
        t1 += t1_rate(
            noise_op=qubit.d_hamiltonian_d_flux_operator(),
            spectral_density=spectral_density_fbl(eigvals, True, M, R_0, T)
            + spectral_density_fbl(eigvals, False, M, R_0, T),
            eigvecs=eigvecs,
        )

    if "t1_charge_impedance" in noise_channels:
        t1 += t1_rate(
            noise_op=qubit.n_operator(),
            spectral_density=spectral_density_ci(eigvals, True, R_0, T, R_k)
            + spectral_density_ci(eigvals, False, R_0, T, R_k),
            eigvecs=eigvecs,
        )

    if "t1_inductive" in noise_channels:
        t1 += t1_rate(
            noise_op=qubit.phi_operator(),
            spectral_density=spectral_density_ind(qubit=qubit, eigvals=eigvals, plus_minus_omega=True, T=T)
            + spectral_density_ind(qubit=qubit, eigvals=eigvals, plus_minus_omega=False, T=T),
            eigvecs=eigvecs,
        )

    if "t1_quasiparticle_tunneling" in noise_channels:
        t1 += t1_rate(
            noise_op=qubit.sin_phi_operator(x=0.5 * (2 * np.pi * qubit.flux)),
            spectral_density=spectral_density_qt(qubit=qubit, eigvals=eigvals, plus_minus_omega=True, T=T)
            + spectral_density_qt(qubit=qubit, eigvals=eigvals, plus_minus_omega=False, T=T),
            eigvecs=eigvecs,
        )

    return t1


def effective_tphi_rate(
    qubit,
    eigvecs: torch.Tensor,
    noise_channels: Union[str, List[str]],
    A_cc: float = NOISE_PARAMS["A_cc"],
    A_flux: float = NOISE_PARAMS["A_flux"],
    A_ng: float = NOISE_PARAMS["A_ng"],
) -> torch.Tensor:
    tphi = torch.zeros([1, 1], dtype=torch.double)

    # tphi_1_over_f_flux
    if "tphi_1_over_f_flux" in noise_channels:
        tphi += tphi_rate(A_flux, qubit.d_hamiltonian_d_flux_operator(), eigvecs=eigvecs)

    # tphi_1_over_f_cc
    if "tphi_1_over_f_cc" in noise_channels:
        tphi += tphi_rate(A_cc, qubit.d_hamiltonian_d_EJ_operator(), eigvecs=eigvecs)

    # tphi_1_over_f_ng
    if "tphi_1_over_f_ng" in noise_channels:
        tphi += tphi_rate(A_ng, qubit.d_hamiltonian_d_ng_operator(), eigvecs=eigvecs)

    return tphi


# T2 RATE
def t2_rate(qubit, eigvecs, eigvals) -> torch.Tensor:
    t1_noise_channels = qubit.t1_supported_noise_channels()
    tphi_noise_channels = qubit.tphi_supported_noise_channels()
    t1_rate = effective_t1_rate(qubit=qubit, eigvals=eigvals, eigvecs=eigvecs, noise_channels=t1_noise_channels)
    tphi_rate = effective_tphi_rate(qubit=qubit, eigvecs=eigvecs, noise_channels=tphi_noise_channels)
    return 0.5 * t1_rate + tphi_rate


# SPECTRAL DENSITIES FOR T1


# CAPACITIVE
def q_cap_fun(eigvals) -> torch.Tensor:
    return 1e6 * torch.pow((2 * np.pi * 6e9 / torch.abs(omega(eigvals) * (1e9))), 0.7)


def spectral_density_cap(qubit, eigvals, plus_minus_omega: bool, T: float = NOISE_PARAMS["T"]):
    omega_for_calc = omega(eigvals) if plus_minus_omega else -omega(eigvals)
    therm_ratio = calc_therm_ratio(omega_for_calc, T)
    s = (
        2
        * 8
        * qubit.EC
        / q_cap_fun(eigvals)
        * (1 / torch.tanh(0.5 * torch.abs(therm_ratio)))
        / (1 + torch.exp(-therm_ratio))
    )
    s *= 2 * np.pi
    return s


# FLUX BIAS LINE
def spectral_density_fbl(
    eigvals,
    plus_minus_omega: bool,
    M: float = NOISE_PARAMS["M"],
    R_0: float = NOISE_PARAMS["R_0"],
    T: float = NOISE_PARAMS["T"],
):
    omega_for_calc = omega(eigvals) if plus_minus_omega else -omega(eigvals)
    therm_ratio = calc_therm_ratio(omega_for_calc, T)
    s = (
        2
        * (2 * np.pi) ** 2
        * M**2
        * (omega(eigvals) * 1e9)
        * sp.constants.hbar
        / R_0
        * (1 / torch.tanh(0.5 * therm_ratio))
        / (1 + torch.exp(-therm_ratio))
    )

    # Unsure why an extra factor of 1e9 is needed?
    return s * 1e9


# CHARGE IMPEDANCE
def spectral_density_ci(
    eigvals,
    plus_minus_omega: bool,
    R_0: float = NOISE_PARAMS["R_0"],
    T: float = NOISE_PARAMS["T"],
    R_k: float = NOISE_PARAMS["R_k"],
):
    # Note, our definition of Q_c is different from Zhang et al (2020) by a
    # factor of 2

    omega_for_calc = omega(eigvals) if plus_minus_omega else -omega(eigvals)

    Q_c = R_k / (8 * np.pi * complex(R_0).real)
    therm_ratio = calc_therm_ratio(omega_for_calc, T)
    s = 2 * (omega_for_calc) / Q_c * (1 / torch.tanh(0.5 * therm_ratio)) / (1 + torch.exp(-therm_ratio))
    return s


# INDUCTIVE
def q_ind_fun(eigvals, plus_minus_omega: bool, T: float = NOISE_PARAMS["T"]):
    omega_for_calc = omega(eigvals) if plus_minus_omega else -omega(eigvals)
    therm_ratio = abs(calc_therm_ratio(omega_for_calc, T))
    therm_ratio_500MHz = calc_therm_ratio(omega=torch.tensor(2 * np.pi * 500e6) / 1e9, T=T)

    return (
        500e6
        * (
            torch.special.scaled_modified_bessel_k0(1 / 2 * therm_ratio_500MHz)
            * torch.sinh(1 / 2 * therm_ratio_500MHz)
            / torch.exp(1 / 2 * therm_ratio_500MHz)
        )
        / (
            torch.special.scaled_modified_bessel_k0(1 / 2 * therm_ratio)
            * torch.sinh(1 / 2 * therm_ratio)
            / torch.exp(1 / 2 * therm_ratio)
        )
    )  ##multiplying each through by the torch.exp(1 / 2 * therm_ratio) seems to work but is different to scqubits?


def spectral_density_ind(qubit, eigvals, plus_minus_omega: bool, T: float = NOISE_PARAMS["T"]):
    omega_for_calc = omega(eigvals) if plus_minus_omega else -omega(eigvals)
    therm_ratio = calc_therm_ratio(omega_for_calc, T)

    s = (
        2
        * qubit.EL
        / q_ind_fun(eigvals, T)
        * (1 / torch.tanh(0.5 * torch.abs(therm_ratio)))
        / (1 + torch.exp(-therm_ratio))
    )
    s *= 2 * np.pi  # We assume that system energies are given in units of frequency
    return s


# QUASIPARTICLE TUNNELLING
def y_qp_fun(
    qubit,
    eigvals,
    T: float = NOISE_PARAMS["T"],
    R_k: float = NOISE_PARAMS["R_k"],
    Delta: float = NOISE_PARAMS["Delta"],
    x_qp: float = NOISE_PARAMS["x_qp"],
):
    Delta_in_Hz = Delta * sp.constants.e / sp.constants.h
    omega_in_Hz = torch.abs(omega(eigvals)) * 1e9 / (2 * np.pi)
    EJ_in_Hz = qubit.EJ * 1e9

    therm_ratio = calc_therm_ratio(torch.abs(omega(eigvals)), T)
    re_y_qp = (
        np.sqrt(2 / np.pi)
        * (8 / R_k)
        * (EJ_in_Hz / Delta_in_Hz)
        * (2 * Delta_in_Hz / omega_in_Hz) ** (3 / 2)
        * x_qp
        * torch.sqrt(1 / 2 * therm_ratio)
        * torch.special.scaled_modified_bessel_k0(1 / 2 * torch.abs(therm_ratio))
        * torch.sinh(1 / 2 * therm_ratio)
        / torch.exp(1 / 2 * torch.abs(therm_ratio))
    )

    return re_y_qp


def spectral_density_qt(
    qubit,
    eigvals,
    plus_minus_omega: bool,
    T: float = NOISE_PARAMS["T"],
):
    omega_for_calc = omega(eigvals) if plus_minus_omega else -omega(eigvals)
    therm_ratio = calc_therm_ratio(omega_for_calc, T)
    return (
        2
        * omega_for_calc
        * complex(y_qp_fun(qubit, eigvals)).real
        * (1 / torch.tanh(0.5 * therm_ratio))
        / (1 + torch.exp(-therm_ratio))
    )
