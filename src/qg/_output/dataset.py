"""
FR-dataset extraction helpers used at solve time.

For an FR run we want to save, beyond the usual outputs, enough information
for downstream Pi_FF post-processing:
  - the static obstacle mask chi_obs (FR grid, smoothed as the solver sees it),
  - the static sponge mask chi_sponge (FR grid),
  - all run parameters needed to rebuild the operator kernels later,
  - per-snapshot vorticity (we already save the 4-channel state for visuals;
    omega alone is enough for post-processing).

These functions are scenario-agnostic: missing terms return None.
"""

import numpy as np
import torch
import yaml

from qg.solver.opt.operator.obstacle import solve_mask as _wrap_obstacle_mask
from qg._input.sources.bc import Region


def extract_obstacle_mask(param, grid, derivative, state):
    """
    Return chi_obs as a torch tensor on the FR grid, or None if no obstacle is active.

    Replicates the wrapping that `define_explicit_operator` does, so the returned
    chi matches what the Brinkman patch actually saw during the run (including
    the 3x3 Gaussian smoothing in solver/opt/operator/obstacle.py).
    """
    if not (param.pde.penalty > 0 and param.mask is not None):
        return None
    # param.mask after vc.validate(...).solve() is a callable
    # (from qg._input.mask.mask.solve_mask). The second-stage wrapper builds the
    # smoothed (chi, vel) tuple, callable as (op, state) -> (chi, vel).
    wrapper = _wrap_obstacle_mask(param.mask, grid, derivative)
    chi, _vel = wrapper(None, state)  # op=None is fine for static masks
    return chi.detach().cpu()


def extract_sponge_mask(raw_bc_cfg, grid):
    """
    Return the sponge ramp tensor(s) as a dict, or None if no sponge is active.

    Dispatches on the original BC function name from the raw config dict
    (raw_bc_cfg = self.raw_param['bc']). Possible return:
       - {'ramp': T}              for vorticity-style BCs (single ramp)
       - {'v1': T1, 'v2': T2, 'ramp': T}  for diffuse-style BCs (triple)
       - None                      for periodic / unknown
    """
    if raw_bc_cfg is None:
        return None
    fn = raw_bc_cfg.get('function')
    if fn in (None, 'periodic'):
        return None

    _min  = raw_bc_cfg.get('_min', 0.0)
    _max  = raw_bc_cfg.get('_max', 1.0)
    width = raw_bc_cfg.get('width', 0.05)

    if fn == 'const-outlet-vorticity-r':
        ramp = Region.outlet_mask_r(grid, _min, _max, width, type='single')
        return {'ramp': ramp.detach().cpu()}
    if fn == 'const-outlet-vorticity-rtd':
        ramp = Region.outlet_mask_rtd(grid, _min, _max, width, type='single')
        return {'ramp': ramp.detach().cpu()}
    if fn == 'const-outlet-diffuse-r':
        v1, v2, ramp = Region.outlet_mask_r(grid, _min, _max, width, type='double')
        return {'v1': v1.detach().cpu(), 'v2': v2.detach().cpu(),
                'ramp': ramp.detach().cpu()}
    if fn == 'const-outlet-diffuse-rtd':
        v1, v2, ramp = Region.outlet_mask_rtd(grid, _min, _max, width, type='triple')
        return {'v1': v1.detach().cpu(), 'v2': v2.detach().cpu(),
                'ramp': ramp.detach().cpu()}
    # Unknown variant — bail out rather than silently miss something
    return None


def build_dataset_npz(omega_FR, chi_obs, sponge_masks, save_index_count, dt, save_rate):
    """
    Pack the FR dataset into a flat dict suitable for np.savez_compressed.

    omega_FR: (B, T_save, Ny, Nx) numpy array, vorticity in physical space
    chi_obs:  (1, Ny, Nx) numpy array, or None
    sponge_masks: dict of named (1, Ny, Nx) arrays, or None
    """
    out = {'omega_FR': omega_FR.astype(np.float32),
           'times': np.arange(save_index_count, dtype=np.float64) * (save_rate * dt)}
    if chi_obs is not None:
        out['chi_obs'] = chi_obs.numpy().astype(np.float32)
    if sponge_masks is not None:
        for k, t in sponge_masks.items():
            out[f'chi_sponge_{k}'] = t.numpy().astype(np.float32)
    return out


def write_metadata_yaml(path, raw_param):
    """Dump the raw (pre-validation) config snapshot as a YAML sidecar."""
    with open(path, 'w') as f:
        yaml.safe_dump(raw_param, f, sort_keys=False, default_flow_style=False)
