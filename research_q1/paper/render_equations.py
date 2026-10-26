"""Render publication equations as high-resolution PNGs for the Word copy."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/flight-fuel-q1-matplotlib")

import matplotlib.pyplot as plt


EQUATIONS = {
    "residual": r"$\widehat{F}_i=F_{\mathrm{phys},i}+g_{\theta}(x_i)$",
    "breguet": r"$F_{\mathrm{cruise}}=m_0\left[1-\exp\left(-\frac{g c_T R}{V(L/D)}\right)\right]$",
    "score": r"$s_i=\frac{|F_i-\widehat{F}_i|}{\max(\widehat{\sigma}(x_i),\varepsilon)}$",
    "interval": r"$C_i=[\max(0,\widehat{F}_i-q_{g(i)}\widehat{\sigma}_i),\ \min(T_i,\widehat{F}_i+q_{g(i)}\widehat{\sigma}_i)]$",
    "selection": r"$\max_{z\in\{0,1\}^{N}}\sum_i u_i z_i\quad \mathrm{s.t.}\quad\sum_i r_i z_i\leq B,\quad\sum_i z_i\geq\lceil mN\rceil$",
    "robust": r"$\sum_i r_i z_i+\Gamma\theta+\sum_i p_i\leq B,\quad p_i\geq d_i z_i-\theta,\quad p_i,\theta\geq 0$",
}


def main() -> None:
    directory = Path(__file__).resolve().parent / "generated/equations"
    directory.mkdir(parents=True, exist_ok=True)
    for name, expression in EQUATIONS.items():
        fig = plt.figure(figsize=(8, .6), dpi=300)
        fig.text(.5, .5, expression, ha="center", va="center", fontsize=20, color="black")
        fig.savefig(directory / f"{name}.png", transparent=True, bbox_inches="tight", pad_inches=.04)
        plt.close(fig)


if __name__ == "__main__":
    main()
