import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from qg.solver.opt.basis import _state, to_spectral, to_physical

import qg.config as vc

from qg.solver.grid.cartesian import CartesianGrid
from qg.solver.opt.derivative import Derivative
from qg.solver.opt.operator import ImplicitLinearOperator, define_explicit_operator
from qg.solver.integrator import Integrator

from qg.solver.opt.operator.jacobian import advection_uv

from qg._output.dataset import (
    extract_obstacle_mask,
    extract_sponge_mask,
    build_dataset_npz,
    write_metadata_yaml,
)

import os
import logging
import jpcm.draw as draw

from omegaconf import DictConfig, OmegaConf

class QG():
    def __init__(self, param,
                 grid = CartesianGrid,
                 derivative = Derivative,
                 implicit_linear_operator = ImplicitLinearOperator,
                 explicit_sources = [],
                 logger = logging.getLogger(__name__)):
        
        # DictConfig already supports attribute access, so just use it directly
        # No need to convert - validate() expects object with attributes
        
        # Snapshot the raw (pre-validation) config BEFORE vc.validate(...).solve()
        # mutates it in place (turns the bc/mask/forcing dicts into callables).
        # The FR-dataset save needs the dicts for the metadata sidecar and to know
        # which sponge-mask variant to rebuild.
        self.raw_param = OmegaConf.to_container(param, resolve=True) if isinstance(param, DictConfig) else dict(param)

        param = vc.validate(param).solve()
        self.param = param
        self.logger = logger
        self.logger.addHandler(logging.StreamHandler())
        
        self.grid = grid(**param.grid)
        self.derivative = derivative(self.grid).to(self.grid.device)
        self.implicit_linear_operator = implicit_linear_operator(self.grid, self.derivative, param.pde)
        self.operator = define_explicit_operator(param, self.grid, self.derivative, self.logger,
                                        args=(param.time.dt, self.grid, self.derivative, param.pde),
                                        sources=explicit_sources) 
        
        self.flow = param.flow # puv bc
        
        self.int = Integrator(param.integrator)
        
        self.dt = param.time.dt
        
        # Select step implementation based on split_bc at init time
        if param.integrator.split_bc:
            step_impl = self._step_with_split
        else:
            step_impl = self._step_without_split
        
        # try:
        #     self.step = torch.compile(step_impl)
        # except Exception as e:
        #     self.logger.warn(f"Failed to compile stepper with exception {e}")
        self.step = step_impl
        
        self.logger.info(f"Initialized QG model with {self.grid.Nx}x{self.grid.Ny} grid on {self.grid.device}")

    def _step_with_split(self, state):
        state.qh = self.int.ex(state.qh, state, state.dt, self.operator.split_source)
        state.qh = self.int.imex(state.qh, state, state.dt, self.operator.source, self.implicit_linear_operator)
        state.update_t()

    def _step_without_split(self, state):
        state.qh = self.int.imex(state.qh, state, state.dt, self.operator.source, self.implicit_linear_operator)
        state.update_t()
   
        # potential flow velocity step
        # state.x_adv, state.y_adv = advection_uv(self.operator, state)
        
        # print(torch.max(to_physical(explicit_source)), torch.min(to_physical(explicit_source)))
        # print(torch.max(to_physical(state.qh)), torch.min(to_physical(state.qh)))
        
        # state.update_potential_flow() # also potential_flow
        # state.dt = self.dt # Not sure if this is necessary, need to think about adaptive time stepping TODO

    def init(self):  
        return _state(self.param.ic(self.grid, self.derivative), self.dt, self.flow, self.derivative) # In spectral space
          
    def _run(self, prof=None, nan_check=False, lim_check=-1):
        save_rate = self.param.time.save_rate
        steps = int(self.param.time.T / self.dt)  # Number of time steps
        
        state = self.init()
        
        # print(torch.max(to_physical(state.qh)), torch.min(to_physical(state.qh)))
        
        B = state.qh.shape[0]  # Number of batches
        solution = torch.zeros([B, int(steps/save_rate)+1, 4, self.grid.Ny, self.grid.Nx])
        
        for it in tqdm(range(steps)):
            self.step(state)            
            
            if (it+1) % save_rate == 0:
                save_index = (it + 1) // save_rate
                solution[:, save_index, ...] = state.out() # B T C H W
            
            if prof is not None:
                prof.step()  # Step the profiler
                
            if nan_check and torch.isnan(state.qh).any():
                self.logger.warning(f"NaN detected at iteration {it}")
                return solution[:,:save_index,...]  # Return what we have so far
                break
            
            if lim_check > 0 and torch.abs(state.qh.real).mean() > lim_check:  # Arbitrary large value
                self.logger.warning(f"Value overflow detected at iteration {it}")
                return solution[:,:save_index,...]  # Return what we have so far
                break
            
        solution[:, -1, ...] = state.out() # B T C H W
                
        return solution
    
    def solve(self, save_path, name='DNS', clamp=0.3, nan_check=False, lim_check=-1): # for direct user call
        if hasattr(self.param, 'profile') and self.param.profile:
            self.logger.info(f"Profiling enabled.")
            from torch.profiler import profile, ProfilerActivity, record_function
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],record_shapes=True, with_stack=True) as prof:
                with record_function("_run"):
                    solution_torch = self._run(prof, nan_check=nan_check, lim_check=lim_check)            
            print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
            prof.export_chrome_trace(os.path.join(save_path, f'{name}_trace.json'))
            self.logger.info(f"Profile trace saved at {os.path.join(save_path, f'{name}_trace.json')}")
        else:
            solution_torch = self._run(nan_check=nan_check, lim_check=lim_check)
        solution = solution_torch.cpu().numpy()
        self.logger.info(f"Simulation complete.")
            
        
        np.save(os.path.join(save_path,f'{name}.npy'), solution)
        self.logger.info(f"Simulation saved at {save_path}")

        # ----------------------------------------------------------------- #
        # FR dataset packed save: omega + static masks + run metadata.      #
        # Consumed by spatial_closure/compute_pi_ff.py (Stage 2, Pi_FF).    #
        # Scenario-agnostic: masks that are not active in this run are      #
        # skipped.  times computed directly (this solver has no save_after, #
        # so t_k = k * save_rate * dt is exact).                            #
        # ----------------------------------------------------------------- #
        omega_FR = solution[:, :, 0, ...]  # (B, T_save, Ny, Nx); channel 0 = vorticity (physical)

        with torch.no_grad():
            dummy_state = self.init()
        chi_obs = extract_obstacle_mask(self.param, self.grid, self.derivative, dummy_state)
        chi_sponge = extract_sponge_mask(
            self.raw_param.get('qg', {}).get('bc', None) or self.raw_param.get('bc', None),
            self.grid)

        T_save = omega_FR.shape[1]
        dataset = build_dataset_npz(omega_FR, chi_obs, chi_sponge,
                                    save_index_count=T_save,
                                    dt=self.param.time.dt,
                                    save_rate=self.param.time.save_rate)
        np.savez_compressed(os.path.join(save_path, f'{name}_FR.npz'), **dataset)
        self.logger.info(f"FR dataset saved: {name}_FR.npz "
                         f"(omega {omega_FR.shape}, "
                         f"chi_obs={'yes' if chi_obs is not None else 'no'}, "
                         f"chi_sponge={'yes' if chi_sponge is not None else 'no'})")

        write_metadata_yaml(os.path.join(save_path, f'{name}_FR_params.yaml'),
                            self.raw_param)
        self.logger.info(f"FR run params saved at {name}_FR_params.yaml")
        
        # select a couple batches for visualization (permute 0,1 axes)
        solution_b = np.transpose(solution[0:4,:,:1,...],(1,0,2,3,4))  # T (selected_B) C H W

        draw.mp4(os.path.join(save_path,f'{name}.mp4'), solution_b,
                   fps=self.param.fps, triplet=True)
        draw.mp4(os.path.join(save_path,f'{name}_clamped.mp4'), solution_b,
                   fps=self.param.fps, triplet=True, clamp=clamp)
        draw.mp4(os.path.join(save_path,f'{name}_seismic.mp4'), solution_b,
                   fps=self.param.fps, triplet=True, cmap='seismic', clamp=clamp)  
        
        # draw.mp4(os.path.join(save_path,'DNS.mp4'), solution_b,
        #            fps=20, triplet=False, mn = [4,1])
        # draw.mp4(os.path.join(save_path,'DNS_clamped.mp4'), solution_b,
        #            fps=20, triplet=False, mn = [4,1], clamp=0.3)
        # draw.mp4(os.path.join(save_path,'DNS_seismic.mp4'), solution_b,
        #            fps=20, triplet=False, mn = [4,1], cmap='seismic', clamp=0.3)    
        
        # # make streamlines from the vorticity field
        # draw.streamlines(os.path.join(save_path,f'{name}_streamlines.mp4'), solution_b[:,:,1,...], -solution_b[:,:,2,...], 
        #                  fps=self.param.fps) # u, -v
        
        
        
        
           
        self.logger.info(f"Videos saved.")
        
        return solution_torch

    def nn_step(self, u, dt=None):
        if dt is None:
            dt = self.dt
        qh = to_spectral(u) # assumes B H W, vorticity only
        state = _state(qh, dt, self.derivative) # In spectral space
        self.step(state)
        return state._out()[:,None,None,...]  # B H W