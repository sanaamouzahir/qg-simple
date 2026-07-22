"""
LES downscaling filter (Eq. 5 of arXiv 2508.06678).

Composite filter:  H_full = interpolate_dx_FF  o  H_cutoff  o  H_Gaussian
  - H_Gaussian: spectral Gaussian with bandwidth (width * dx_FR), exp(-k**2 (width*dx)**2 / 6)
  - H_cutoff:   sharp axis-wise cutoff at |kx|, |ky| < pi/(scale*dx_FR)
  - interpolate: spatial coarsening via 2D average pooling with stride=scale

The two parameters are now independent:
  * `scale` is the integer downsampling factor (paper's delta).
  * `width` is the Gaussian bandwidth multiplier (paper uses 1.5).
The paper's convention: filter has Gaussian width = 1.5*dx_FR but coarse-grid
ratio = 2 (scale=2). Default behavior here matches that: width defaults to
the paper's 1.5 unless explicitly given.

The input is on the FR grid; the output lives on a grid with shape (Ny/scale, Nx/scale).
"""

import torch
import torch.nn.functional as F
from qg.solver.opt.basis import to_physical, to_spectral


class LESFilter:
    """
    Pre-computes Gaussian + cutoff masks once from the FR grid + derivative.
    Reusable across snapshots.

    Parameters
    ----------
    grid : CartesianGrid
        FR grid.
    derivative : Derivative
        Derivative built from the FR grid.
    scale : int
        Integer downsampling factor (FR Nx/Ny is divided by this for the LES grid).
    width : float, optional
        Gaussian bandwidth multiplier. Default: 1.5 (the paper's value).

    Use `from_spectral` if the input is already in spectral space (rfft layout),
    `from_physical` if the input is a physical-space tensor on the FR grid.

    Set `output='physical'` for a downsampled physical field of shape (..., Ny/scale, Nx/scale),
    or `output='spectral'` for its rfft on the coarse grid.
    """

    def __init__(self, grid, derivative, scale, width=1.5):
        if grid.Nx % scale != 0 or grid.Ny % scale != 0:
            raise ValueError(f"FR grid ({grid.Nx},{grid.Ny}) must be divisible by scale={scale}.")
        self.grid = grid
        self.derivative = derivative
        self.scale = int(scale)
        self.width = float(width)

        # Gaussian-bandwidth physical length (paper Sect III, Eq. 5):
        #   sigma = width * dx_FR
        gaussian_sigma = self.width * grid.dx
        # Spectral Gaussian:  exp(-k**2 sigma**2 / 6).
        # NOTE: in qg-simple, derivative.laplacian = -ksq (negative); we use +ksq directly.
        self._gaussian = torch.exp(-grid.ksq * (gaussian_sigma ** 2) / 6)

        # Sharp cutoff at coarse-grid Nyquist (set by `scale`, not `width`):
        #   |k| < pi / dx_LES,  where dx_LES = scale * dx_FR
        dx_LES = self.scale * grid.dx
        kcut = torch.pi / dx_LES
        self._cutoff_x = (grid.kx.abs() < kcut)
        self._cutoff_y = (grid.ky.abs() < kcut)

    def from_spectral(self, qh, output='physical'):
        """Filter a spectral-space tensor on the FR grid. qh shape: (..., Ny, Nx//2+1)."""
        # Stage 1: Gaussian (uses width)
        qh_filt = self._gaussian * qh
        # Stage 2: sharp cutoff (uses scale)
        qh_filt = self._cutoff_y * qh_filt
        qh_filt = self._cutoff_x * qh_filt
        # Stage 3: inverse FFT then average-pool to coarse grid (uses scale)
        q_FR = to_physical(qh_filt)
        q_coarse = self._avg_pool(q_FR)
        if output == 'physical':
            return q_coarse
        elif output == 'spectral':
            return to_spectral(q_coarse)
        else:
            raise ValueError(f"output must be 'physical' or 'spectral', got {output!r}")

    def from_physical(self, q, output='physical'):
        """Filter a physical-space tensor on the FR grid. q shape: (..., Ny, Nx)."""
        return self.from_spectral(to_spectral(q), output=output)

    def _avg_pool(self, q_FR):
        """
        Average-pool over the last two dims by `scale`, preserving any leading dims.
        Accepts shapes (Ny, Nx), (B, Ny, Nx), (B, T, Ny, Nx), etc.
        """
        s = self.scale
        if q_FR.dim() == 2:
            return F.avg_pool2d(q_FR[None, None], kernel_size=s, stride=s)[0, 0]
        if q_FR.dim() == 3:
            return F.avg_pool2d(q_FR[:, None], kernel_size=s, stride=s)[:, 0]
        leading = q_FR.shape[:-2]
        flat = q_FR.reshape(-1, 1, *q_FR.shape[-2:])
        pooled = F.avg_pool2d(flat, kernel_size=s, stride=s)
        return pooled.reshape(*leading, *pooled.shape[-2:])
