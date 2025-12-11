# script for diffusion protocols
import torch
import pickle
import numpy as np
import os
import logging

from scipy.spatial.transform import Rotation as scipy_R

from rfdiff.util import rigid_from_3_points

from rfdiff.util_module import ComputeAllAtomCoords

from rfdiff import igso3
import time

from mytest.DDSM import DirichletDiffuser

torch.set_printoptions(sci_mode=False)


def get_beta_schedule(T, b0, bT, schedule_type, schedule_params={}, inference=False):
    """
    Given a noise schedule type, create the beta schedule
    """
    assert schedule_type in ["linear"]

    # Adjust b0 and bT if T is not 200
    # This is a good approximation, with the beta correction below, unless T is very small
    assert T >= 15, "With discrete time and T < 15, the schedule is badly approximated"
    b0 *= 200 / T
    bT *= 200 / T

    # linear noise schedule
    if schedule_type == "linear":
        schedule = torch.linspace(b0, bT, T)

    else:
        raise NotImplementedError(f"Schedule of type {schedule_type} not implemented.")

    # get alphabar_t for convenience
    alpha_schedule = 1 - schedule
    alphabar_t_schedule = torch.cumprod(alpha_schedule, dim=0)

    if inference:
        print(
            f"With this beta schedule ({schedule_type} schedule, beta_0 = {round(b0, 3)}, beta_T = {round(bT,3)}), alpha_bar_T = {alphabar_t_schedule[-1]}"
        )

    return schedule, alpha_schedule, alphabar_t_schedule

class EuclideanDiffuser:
    """Class for diffusing points in 3D with native batch support."""

    def __init__(
        self,
        T,
        b_0,
        b_T,
        schedule_type="linear",
        schedule_kwargs={},
    ):
        self.T = T

        # make noise/beta schedule
        (
            self.beta_schedule,
            self.alpha_schedule,
            self.alphabar_schedule,
        ) = get_beta_schedule(T, b_0, b_T, schedule_type, **schedule_kwargs)

    def diffuse_translations(self, xyz, diffusion_mask=None, var_scale=1):
        """
        Diffuse translations with batch support.
        
        Args:
            xyz: coordinates of shape (..., L, 3, 3) where ... are batch dims
            diffusion_mask: mask of shape (..., L) or (L,) 
            var_scale: variance scaling factor
            
        Returns:
            Tuple of (diffused_coords, deltas) with shapes (..., T, L, 3, 3) and (..., T, L, 3)
        """
        return self.apply_kernel_recursive(xyz, diffusion_mask, var_scale)

    def apply_kernel(self, x, t, diffusion_mask=None, var_scale=1):
        """
        Apply noising kernel with batch support.

        Args:
            x: coordinates of shape (..., L, 3, 3) where ... are batch dims
            t: timestep (int)
            diffusion_mask: mask of shape (..., L) or (L,)
            var_scale: variance scaling factor

        Returns:
            Tuple of (noised_coords, delta) with shapes (..., L, 3, 3) and (..., L, 3)
        """
        t_idx = t - 1  # bring from 1-indexed to 0-indexed

        assert len(x.shape) >= 3, "Input must have at least 3 dimensions (L, 3, 3)"
        
        *batch_dims, L, _, _ = x.shape

        # Extract C-alpha coordinates
        ca_xyz = x[..., 1, :]  # (..., L, 3)
        b_t = self.beta_schedule[t_idx]
        # Get noise parameters
        mean = torch.sqrt(1 - b_t) * ca_xyz
        var = torch.ones_like(ca_xyz) * (b_t * var_scale)
        # Sample noise
        sampled_crds = torch.normal(mean, torch.sqrt(var))
        delta = sampled_crds - ca_xyz
        # Apply diffusion mask
        if diffusion_mask is not None:
            # Handle mask broadcasting
            if diffusion_mask.dim() == 1:  # Shape (L,)
                # Broadcast to match batch dimensions
                for _ in range(len(batch_dims)):
                    diffusion_mask = diffusion_mask.unsqueeze(0)
            
            # Expand mask to match delta shape
            mask_expanded = diffusion_mask.unsqueeze(-1).expand_as(delta)
            delta = delta * (~mask_expanded).float()

        # Apply delta to coordinates
        out_crds = x + delta.unsqueeze(-2)  # (..., L, 1, 3) -> (..., L, 3, 3)

        return out_crds, delta

    def apply_kernel_recursive(self, xyz, diffusion_mask=None, var_scale=1):
        """
        Repeatedly apply kernel T times with batch support.
        
        Args:
            xyz: coordinates of shape (..., L, 3, 3)
            diffusion_mask: mask of shape (..., L) or (L,)
            var_scale: variance scaling factor
            
        Returns:
            Tuple of (coords_stack, deltas_stack) with shapes (..., T, L, 3, 3) and (..., T, L, 3)
        """
        original_shape = xyz.shape
        
        bb_stack = []
        T_stack = []

        cur_xyz = torch.clone(xyz)

        for t in range(1, self.T + 1):
            cur_xyz, cur_T = self.apply_kernel(
                cur_xyz, t, var_scale=var_scale, diffusion_mask=diffusion_mask
            )
            bb_stack.append(cur_xyz)
            T_stack.append(cur_T)

        # Stack along new time dimension
        # For input shape (..., L, 3, 3), output should be (..., T, L, 3, 3)
        coords_stack = torch.stack(bb_stack, dim=-4)  # Insert T dimension before L
        deltas_stack = torch.stack(T_stack, dim=-3)   # Insert T dimension before L

        return coords_stack, deltas_stack


def write_pkl(save_path: str, pkl_data):
    """Serialize data into a pickle file."""
    with open(save_path, "wb") as handle:
        pickle.dump(pkl_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pkl(read_path: str, verbose=False):
    """Read data from a pickle file."""
    with open(read_path, "rb") as handle:
        try:
            return pickle.load(handle)
        except Exception as e:
            if verbose:
                print(f"Failed to read {read_path}")
            raise (e)


class IGSO3:
    """
    Class for IGSO3 diffusion with native batch support.
    """

    def __init__(
        self,
        *,
        T,
        min_sigma,
        max_sigma,
        min_b,
        max_b,
        cache_dir,
        num_omega=1000,
        schedule="linear",
        L=2000,
    ):
        self._log = logging.getLogger(__name__)

        self.T = T
        self.schedule = schedule
        self.cache_dir = cache_dir
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.min_b = min_b
        self.max_b = max_b

        # Set max_sigma after defining sigma method for linear schedule
        if self.schedule == "linear":
            # For linear schedule, max_sigma is computed from sigma(1.0)
            # We'll compute it after defining the sigma method
            pass
        
        self.num_omega = num_omega
        self.num_sigma = 500
        # Calculate igso3 values.
        self.L = L  # truncation level
        
        # Now compute max_sigma for linear schedule
        if self.schedule == "linear":
            self.max_sigma = self.sigma(1.0)
            
        self.igso3_vals = self._calc_igso3_vals(L=L)
        self.step_size = 1 / self.T

    def sigma(self, t):
        """
        Extract \sigma(t) corresponding to chosen sigma schedule.

        Args:
            t: time between 0 and 1 (can be scalar, numpy array, or torch tensor)
        """
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32)
        
        if torch.any(t < 0) or torch.any(t > 1):
            raise ValueError(f"Invalid t={t}, must be in [0, 1]")
            
        if self.schedule == "exponential":
            sigma = t * np.log10(self.max_sigma) + (1 - t) * np.log10(self.min_sigma)
            return 10**sigma
        elif self.schedule == "linear":  # Variance exploding analogue of Ho schedule
            # add self.min_sigma for stability
            return (
                self.min_sigma
                + t * self.min_b
                + (1 / 2) * (t**2) * (self.max_b - self.min_b)
            )
        else:
            raise ValueError(f"Unrecognized schedule {self.schedule}")

    def g(self, t):
        """
        g returns the drift coefficient at time t

        since
            sigma(t)^2 := \int_0^t g(s)^2 ds,
        for arbitrary sigma(t) we invert this relationship to compute
            g(t) = sqrt(d/dt sigma(t)^2).

        Args:
            t: scalar time between 0 and 1

        Returns:
            drift coefficient as a scalar.
        """
        t = torch.tensor(t, requires_grad=True, dtype=torch.float32)
        sigma_sqr = self.sigma(t) ** 2
        grads = torch.autograd.grad(sigma_sqr.sum(), t)[0]
        return torch.sqrt(grads)

    def _calc_igso3_vals(self, L=2000):
        """_calc_igso3_vals computes numerical approximations to the
        relevant analytically intractable functionals of the igso3
        distribution.

        The calculated values are cached, or loaded from cache if they already
        exist.

        Args:
            L: truncation level for power series expansion of the pdf.
        """
        replace_period = lambda x: str(x).replace(".", "_")
        if self.schedule == "linear":
            cache_fname = os.path.join(
                self.cache_dir,
                f"T_{self.T}_omega_{self.num_omega}_min_sigma_{replace_period(self.min_sigma)}"
                + f"_min_b_{replace_period(self.min_b)}_max_b_{replace_period(self.max_b)}_schedule_{self.schedule}.pkl",
            )
        elif self.schedule == "exponential":
            cache_fname = os.path.join(
                self.cache_dir,
                f"T_{self.T}_omega_{self.num_omega}_min_sigma_{replace_period(self.min_sigma)}"
                f"_max_sigma_{replace_period(self.max_sigma)}_schedule_{self.schedule}.pkl",
            )
        else:
            raise ValueError(f"Unrecognized schedule {self.schedule}")

        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)

        if os.path.exists(cache_fname):
            self._log.info("Using cached IGSO3.")
            igso3_vals = read_pkl(cache_fname)
        else:
            self._log.info("Calculating IGSO3.")
            igso3_vals = igso3.calculate_igso3(
                num_sigma=self.num_sigma,
                min_sigma=self.min_sigma,
                max_sigma=self.max_sigma,
                num_omega=self.num_omega
            )
            write_pkl(cache_fname, igso3_vals)

        return igso3_vals

    @property
    def discrete_sigma(self):
        return self.igso3_vals["discrete_sigma"]

    def sigma_idx(self, sigma):
        """
        Calculates the index for discretized sigma during IGSO(3) initialization.
        """
        if isinstance(sigma, torch.Tensor):
            sigma = sigma.cpu().numpy()
        return np.digitize(sigma, self.discrete_sigma) - 1

    def t_to_idx(self, t):
        """
        Helper function to go from discrete time index t to corresponding sigma_idx.

        Args:
            t: time index (integer between 1 and T)
        """
        if isinstance(t, (list, tuple, np.ndarray)):
            t = np.array(t)
        continuous_t = t / self.T
        return self.sigma_idx(self.sigma(continuous_t))

    def sample(self, ts, n_samples=1):
        """
        sample uses the inverse cdf to sample an angle of rotation from
        IGSO(3)

        Args:
            ts: array of integer time steps to sample from.
            n_samples: number of samples to draw.
        Returns:
            sampled angles of rotation. [len(ts), N]
        """
        if isinstance(ts, (int, float)):
            ts = [ts]
        ts = np.array(ts)
        assert np.all(ts > 0), "assumes one-indexed, not zero indexed"
        
        all_samples = []
        for t in ts:
            sigma_idx = self.t_to_idx(t)
            sample_i = np.interp(
                np.random.rand(n_samples),
                self.igso3_vals["cdf"][sigma_idx],
                self.igso3_vals["discrete_omega"],
            )  # [N, 1]
            all_samples.append(sample_i)
        return np.stack(all_samples, axis=0)

    def sample_vec(self, ts, n_samples=1):
        """sample_vec generates a rotation vector(s) from IGSO(3) at time steps
        ts.

        Return:
            Sampled vector of shape [len(ts), N, 3]
        """
        if isinstance(ts, (int, float)):
            ts = [ts]
        ts = np.array(ts)
        
        x = np.random.randn(len(ts), n_samples, 3)
        x /= np.linalg.norm(x, axis=-1, keepdims=True)
        return x * self.sample(ts, n_samples=n_samples)[..., None]

    def score_norm(self, t, omega):
        """
        score_norm computes the score norm based on the time step and angle
        Args:
            t: integer time step
            omega: angles (scalar or array)
        Return:
            score_norm with same shape as omega
        """
        if isinstance(omega, torch.Tensor):
            omega = omega.cpu().numpy()
        
        sigma_idx = self.t_to_idx(t)
        score_norm_t = np.interp(
            omega,
            self.igso3_vals["discrete_omega"],
            self.igso3_vals["score_norm"][sigma_idx],
        )
        return score_norm_t

    def score_vec(self, ts, vec):
        """score_vec computes the score of the IGSO(3) density as a rotation
        vector. This score vector is in the direction of the sampled vector,
        and has magnitude given by score_norms.

        Args:
            ts: times of shape [T]
            vec: where to compute the score of shape [T, N, 3]
        Returns:
            score vectors of shape [T, N, 3]
        """
        if isinstance(vec, torch.Tensor):
            vec = vec.cpu().numpy()
        if isinstance(ts, (int, float)):
            ts = [ts]
        ts = np.array(ts)
        
        omega = np.linalg.norm(vec, axis=-1)
        all_score_norm = []
        for i, t in enumerate(ts):
            omega_t = omega[i] if omega.ndim > 0 else omega
            sigma_idx = self.t_to_idx(t)
            score_norm_t = np.interp(
                omega_t,
                self.igso3_vals["discrete_omega"],
                self.igso3_vals["score_norm"][sigma_idx],
            )
            if omega_t.ndim > 0:
                score_norm_t = score_norm_t[:, None]
            else:
                score_norm_t = score_norm_t[None]
            all_score_norm.append(score_norm_t)
        score_norm = np.stack(all_score_norm, axis=0)
        return score_norm * vec / np.maximum(omega[..., None], 1e-8)

    def exp_score_norm(self, ts):
        """exp_score_norm returns the expected value of norm of the score for
        IGSO(3) with time parameter ts of shape [T].
        """
        if isinstance(ts, (int, float)):
            ts = [ts]
        sigma_idcs = [self.t_to_idx(t) for t in ts]
        return self.igso3_vals["exp_score_norms"][sigma_idcs]

    def diffuse_frames(self, xyz, t_list=None, diffusion_mask=None):
        """
        Diffuse frames with batch support.

        Args:
            xyz: coordinates of shape (..., L, 3, 3) where ... are batch dims
            t_list: optional list of timesteps to return
            diffusion_mask: mask of shape (..., L) or (L,)
            
        Returns:
            Tuple of (perturbed_coords, R_perturbed) with shapes (..., L, T, 3, 3)
        """
        # Convert to numpy if tensor
        if torch.is_tensor(xyz):
            xyz_np = xyz.cpu().numpy()
            device = xyz.device
            dtype = xyz.dtype
        else:
            xyz_np = xyz
            device = torch.device('cpu')
            dtype = torch.float32

        # Handle batch dimensions
        original_shape = xyz_np.shape
        if len(original_shape) == 3:  # Unbatched: (L, 3, 3)
            xyz_np = xyz_np[None]  # Add batch dim
            batched_input = False
        else:
            batched_input = True

        *batch_dims, L, _, _ = xyz_np.shape
        batch_size = np.prod(batch_dims) if batch_dims else 1

        # Flatten batch dimensions for processing
        xyz_flat = xyz_np.reshape(batch_size, L, 3, 3)

        t = np.arange(self.T) + 1  # 1-indexed

        all_perturbed_coords = []
        all_R_perturbed = []

        # Process each item in batch
        for b in range(batch_size):
            xyz_b = xyz_flat[b]  # (L, 3, 3)
            
            # Get rigid frames
            N = torch.from_numpy(xyz_b[None, :, 0, :])
            Ca = torch.from_numpy(xyz_b[None, :, 1, :])
            C = torch.from_numpy(xyz_b[None, :, 2, :])

            R_true, Ca = rigid_from_3_points(N, Ca, C)
            R_true = R_true[0].numpy()
            Ca = Ca[0].numpy()

            # Sample rotations
            sampled_rots = self.sample_vec(t, n_samples=L)  # [T, L, 3]

            # Apply diffusion mask if provided
            if diffusion_mask is not None:
                if diffusion_mask.ndim == 1:  # Shape (L,)
                    mask_b = diffusion_mask.cpu().numpy() if torch.is_tensor(diffusion_mask) else diffusion_mask
                else:  # Batched mask
                    # Get mask for this batch item
                    batch_idx = np.unravel_index(b, batch_dims) if batch_dims else (0,)
                    mask_b = diffusion_mask[batch_idx]
                    if torch.is_tensor(mask_b):
                        mask_b = mask_b.cpu().numpy()
                
                non_diffusion_mask = (1 - mask_b)[None, :, None]
                sampled_rots = sampled_rots * non_diffusion_mask

            # Apply rotations
            R_sampled = (
                scipy_R.from_rotvec(sampled_rots.reshape(-1, 3))
                .as_matrix()
                .reshape(self.T, L, 3, 3)
            )
            
            R_perturbed = np.einsum("tnij,njk->tnik", R_sampled, R_true)
            perturbed_crds = (
                np.einsum(
                    "tnij,naj->tnai", R_sampled, xyz_b[:, :3, :] - Ca[:, None, ...]
                )
                + Ca[None, :, None]
            )

            all_perturbed_coords.append(perturbed_crds)
            all_R_perturbed.append(R_perturbed)

        # Stack batch results
        all_perturbed_coords = np.stack(all_perturbed_coords, axis=0)  # (B, T, L, 3, 3)
        all_R_perturbed = np.stack(all_R_perturbed, axis=0)  # (B, T, L, 3, 3)

        # Reshape to original batch dimensions
        final_shape = batch_dims + [self.T, L, 3, 3]
        all_perturbed_coords = all_perturbed_coords.reshape(final_shape)
        all_R_perturbed = all_R_perturbed.reshape(final_shape)

        # Handle t_list filtering
        if t_list is not None:
            idx = [i - 1 for i in t_list]
            all_perturbed_coords = all_perturbed_coords[..., idx, :, :, :]
            all_R_perturbed = all_R_perturbed[..., idx, :, :, :]

        # Remove batch dim if input was unbatched
        if not batched_input:
            all_perturbed_coords = all_perturbed_coords[0]
            all_R_perturbed = all_R_perturbed[0]

        # Transpose to match expected output format: (..., L, T, 3, 3)
        if batched_input or len(original_shape) > 3:
            # For batched: (..., T, L, 3, 3) -> (..., L, T, 3, 3)
            all_perturbed_coords = all_perturbed_coords.transpose(
                *list(range(len(batch_dims))), len(batch_dims)+1, len(batch_dims), len(batch_dims)+2, len(batch_dims)+3
            )
            all_R_perturbed = all_R_perturbed.transpose(
                *list(range(len(batch_dims))), len(batch_dims)+1, len(batch_dims), len(batch_dims)+2, len(batch_dims)+3
            )
        else:
            # For unbatched: (T, L, 3, 3) -> (L, T, 3, 3)
            all_perturbed_coords = all_perturbed_coords.transpose(1, 0, 2, 3)
            all_R_perturbed = all_R_perturbed.transpose(1, 0, 2, 3)

        # Convert back to torch tensors
        all_perturbed_coords = torch.from_numpy(all_perturbed_coords).to(device, dtype=dtype)
        all_R_perturbed = torch.from_numpy(all_R_perturbed).to(device, dtype=dtype)

        return all_perturbed_coords, all_R_perturbed

    def reverse_sample_vectorized(
        self, R_t, R_0, t, noise_level, mask=None, return_perturb=False
    ):
        """
        Reverse sample with batch support.
        
        Args:
            R_t: noisy rotations of shape (..., 3, 3) where ... are batch dims
            R_0: predicted clean rotations of shape (..., 3, 3)
            t: timestep (int)
            noise_level: noise scaling
            mask: optional mask of shape (...)
            return_perturb: whether to return perturbation
            
        Returns:
            Sampled rotations of shape (..., 3, 3)
        """
        # Handle batch dimensions
        original_shape = R_t.shape
        if len(original_shape) == 2:  # Unbatched: (3, 3)
            R_t = R_t.unsqueeze(0)
            R_0 = R_0.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
            batched_input = False
        else:
            batched_input = True

        # Compute rotation vector corresponding to prediction
        if not torch.is_tensor(R_0):
            R_0 = torch.tensor(R_0, dtype=torch.float32)
        if not torch.is_tensor(R_t):
            R_t = torch.tensor(R_t, dtype=torch.float32)
        
        # Handle batch matrix operations
        R_0t = torch.matmul(R_t.transpose(-2, -1), R_0)
        
        # Convert to rotation vectors (handle batch)
        batch_shape = R_0t.shape[:-2]
        R_0t_flat = R_0t.view(-1, 3, 3)
        rotvecs_flat = torch.tensor(
            scipy_R.from_matrix(R_0t_flat.cpu().numpy()).as_rotvec()
        ).to(R_0.device)
        R_0t_rotvec = rotvecs_flat.view(*batch_shape, 3)

        # Compute score approximation
        Omega = torch.linalg.norm(R_0t_rotvec, dim=-1).cpu().numpy()
        score_norms = self.score_norm(t, Omega)
        Score_approx = R_0t_rotvec * (
            torch.tensor(score_norms, device=R_0t_rotvec.device) / 
            torch.clamp(torch.tensor(Omega, device=R_0t_rotvec.device), min=1e-8)
        ).unsqueeze(-1)

        # Compute scaling
        continuous_t = t / self.T
        rot_g = self.g(continuous_t).to(Score_approx.device)

        # Sample noise with batch dimensions
        Z = torch.randn(*batch_shape, 3, device=Score_approx.device) * noise_level

        # Compute perturbation
        Delta_r = (rot_g**2) * self.step_size * Score_approx
        Perturb_tangent = Delta_r + rot_g * np.sqrt(self.step_size) * Z

        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(Perturb_tangent)
            Perturb_tangent = Perturb_tangent * (1 - mask_expanded.float())

        # Convert to rotation matrices using igso3.Exp
        perturb_flat = Perturb_tangent.view(-1, 3)
        Perturb_matrices_flat = igso3.Exp(perturb_flat)
        Perturb = Perturb_matrices_flat.view(*batch_shape, 3, 3)

        if return_perturb:
            return Perturb.squeeze(0) if not batched_input else Perturb

        # Compose rotations
        Interp_rot = torch.matmul(Perturb, R_t)

        return Interp_rot.squeeze(0) if not batched_input else Interp_rot

class Diffuser:
    """
    Wrapper for yielding diffused coordinates with native batch support.
    Combines EuclideanDiffuser and IGSO3 for complete pose diffusion.
    """

    def __init__(
        self,
        T,
        b_0,
        b_T,
        min_sigma,
        max_sigma,
        min_b,
        max_b,
        schedule_type,
        so3_schedule_type,
        so3_type,
        crd_scale,
        schedule_kwargs={},
        var_scale=1.0,
        cache_dir=".",
        partial_T=None,
        truncation_level=2000,
        dirichlet_noise_config=None,
        device="cpu",
    ):
        """
        Initialize Diffuser with both translation and rotation diffusion.
        
        Args:
            T: number of diffusion timesteps
            b_0: initial beta for translation diffusion
            b_T: final beta for translation diffusion
            min_sigma: minimum sigma for SO3 diffusion
            max_sigma: maximum sigma for SO3 diffusion
            min_b: minimum b for SO3 diffusion
            max_b: maximum b for SO3 diffusion
            schedule_type: beta schedule type for translation diffusion
            so3_schedule_type: schedule type for SO3 diffusion
            so3_type: type of SO3 diffusion (should be "igso3")
            crd_scale: coordinate scaling factor
            schedule_kwargs: additional kwargs for schedule
            var_scale: variance scaling factor
            cache_dir: directory for caching IGSO3 calculations
            partial_T: partial diffusion steps (if None, use full T)
            truncation_level: truncation level for IGSO3
        """
        self.T = T
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.crd_scale = crd_scale
        self.var_scale = var_scale
        self.cache_dir = cache_dir
        self.partial_T = partial_T

        # Initialize SO3 diffuser
        if so3_type == "igso3":
            self.so3_diffuser = IGSO3(
                T=self.T,
                min_sigma=self.min_sigma,
                max_sigma=self.max_sigma,
                schedule=so3_schedule_type,
                min_b=min_b,
                max_b=max_b,
                cache_dir=self.cache_dir,
                L=truncation_level,
            )
        else:
            raise ValueError(f"Unrecognized so3_type {so3_type}")

        # Initialize Euclidean diffuser
        self.eucl_diffuser = EuclideanDiffuser(
            self.T, b_0, b_T, schedule_type=schedule_type, **schedule_kwargs
        )


        if dirichlet_noise_config is None:
            self.dirichlet_noise_config = {
                'N': 100000,                    # N_noise_trajectories
                'n_time_steps': self.T,        # num_diffusion_timesteps
                'total_time': 1.0,          # total_time_nf
                'sampler_steps': 400,       # sampler_steps_nf
                'speed_balanced': True,     # speed_balanced_nf
                'logspace': False,          # logspace_nf
                'mode': 'path',      # mode_nf
                'order': 200               # order for jacobi polynomials
            }
        else:
            self.dirichlet_noise_config = dirichlet_noise_config

        dirichlet_device = device
        alpha_params = torch.ones(19, device=dirichlet_device)
        beta_params = torch.arange(19, 0, -1, dtype=torch.float, device=dirichlet_device)
        self.dirichlet_diffuser = DirichletDiffuser(
            K=20,
            noise_factory_config =self.dirichlet_noise_config,
            device=dirichlet_device,
            alpha_params=alpha_params,
            beta_params=beta_params,
            cache_dir=cache_dir
        )

        print("Successful diffuser __init__")

    def diffuse_pose(
        self,
        xyz,
        seq=None,
        atom_mask=None,
        include_motif_sidechains=True,
        diffusion_mask=None,
        t_list=None,
    ):
        """
        Diffuse protein poses with batch support.
        
        Args:
            xyz: coordinates of shape (..., L, 3, 3) or (..., L, 27, 3) where ... are batch dims
            seq: sequence (not used in diffusion but kept for compatibility)
            atom_mask: atom mask (not used but kept for compatibility)
            include_motif_sidechains: whether to include motif sidechains
            diffusion_mask: mask indicating which residues to diffuse, shape (..., L)
            t_list: list of timesteps to return (if None, return all)
            
        Returns:
            Tuple of (diffused_coords, xyz_true) where:
            - diffused_coords: shape (..., T, L, 27, 3) if t_list is None, else (..., len(t_list), L, 27, 3)
            - xyz_true: original coordinates for reference
        """
        import time
    
        xyz_bb = xyz[..., :3, :]
        original_shape = xyz_bb.shape
        *batch_dims, L, _, _ = original_shape
        
        # Validate diffusion mask
        if diffusion_mask is None:
            diffusion_mask = torch.zeros(*batch_dims, L, dtype=torch.bool)
        else:
            # Ensure mask has correct shape
            diffusion_mask = ~diffusion_mask
            if diffusion_mask.shape != (*batch_dims, L):
                raise ValueError(f"diffusion_mask shape {diffusion_mask.shape} doesn't match expected {(*batch_dims, L)}")

        # Check for NaN values
        if torch.isnan(xyz_bb).any():
            nan_mask = torch.isnan(xyz_bb).any(dim=-1).any(dim=-1)
            if nan_mask.any():
                xyz_bb.nan_to_num_()
                # print(f"Warning: Found NaN values in {nan_mask.sum().item()} residues")

        # Center structure at origin (prevent information leak)
        xyz_centered = xyz_bb.clone()
        center_mask = ~diffusion_mask
        motif_coords = (xyz_centered[:,:,1,:] * center_mask.unsqueeze(-1).float()).sum(-2, keepdim=True)
        motif_coords = motif_coords / center_mask.sum(dim=-1,keepdim=True).unsqueeze(-1)
        xyz_centered = xyz_centered - motif_coords.unsqueeze(-2)  # Center at motif COM

        xyz_true = xyz_centered.clone()
        xyz_scaled = xyz_centered * self.crd_scale

        # 1. Diffuse translations

        diffused_T, deltas = self.eucl_diffuser.diffuse_translations(
            xyz_scaled, diffusion_mask=diffusion_mask, var_scale=self.var_scale
        )
        # print(f'Time to diffuse coordinates: {time.time()-tick:.3f}s')
        diffused_T /= self.crd_scale
        deltas /= self.crd_scale

        # 2. Diffuse frames (rotations)
        diffused_frame_crds, diffused_frames = self.so3_diffuser.diffuse_frames(
            xyz_scaled, diffusion_mask=diffusion_mask, t_list=None
        )
        diffused_frame_crds /= self.crd_scale
        # print(f'Time to diffuse frames: {time.time()-tick:.3f}s')

        # 3. Combine translations and rotations
        cum_delta = deltas.cumsum(dim=-3)  # Cumulative sum along time dimension
        # Handle batch dimensions for combination
        if len(batch_dims) > 0:
            # For batched input: (..., L, T, 3, 3) -> (..., T, L, 3, 3)
            frame_dims = list(range(len(batch_dims)))
            diffused_frame_crds = diffused_frame_crds.transpose(
                len(batch_dims), len(batch_dims) + 1
            )  # (..., L, T, 3, 3) -> (..., T, L, 3, 3)
            
            # Add cumulative deltas
            diffused_BB = diffused_frame_crds + cum_delta.unsqueeze(-2)  # (..., T, L, 1, 3) -> (..., T, L, 3, 3)
        else:
            # Unbatched: (L, T, 3, 3) -> (T, L, 3, 3)
            diffused_BB = diffused_frame_crds.transpose(0, 1) + cum_delta.unsqueeze(-2)

        #4.5 Diffuse sequence      
        if t_list is None or len(t_list) == 0:
            t_seq = self.T - 1
        else:
            t_seq = t_list[-1] - 1            
        t_seq = max(t_seq, 0)   
        noised_seq, noised_seq_grad = self.dirichlet_diffuser.apply_noise_to_sequence(
            seq_clean=seq,
            chosen_time_idx_scalar=t_seq,
            mask = diffusion_mask,
            return_v=False  # 确保输出在 x 空间 (simplex)
        )

        # 4. Create full atom coordinates
        time_steps = diffused_BB.shape[-4]  # T dimension
        diffused_fa = torch.full((*diffused_BB.shape[:-2], 27, 3),0.0,device=diffused_BB.device) # (..., T, L, 27, 3)
        diffused_fa[..., :3, :] = diffused_BB  # Copy backbone coordinates
        # 5. Add motif sidechains if requested
        if include_motif_sidechains and diffusion_mask.any():
            # For motif residues, use original coordinates for sidechains
            if xyz.shape[-2] == 27:  # If input had sidechain coordinates
                if len(batch_dims) > 0:
                    # Batched processing
                    for t in range(time_steps):
                        diffused_fa[..., t, :, :14, :][diffusion_mask] = xyz[..., :14, :][diffusion_mask]
                else:
                    # Unbatched processing
                    for t in range(time_steps):
                        diffused_fa[t, diffusion_mask, :14, :] = xyz[diffusion_mask, :14, :]

        # 6. Filter by t_list if provided
        if t_list is not None:
            t_idx_list = [t - 1 for t in t_list]  # Convert to 0-indexed
            diffused_fa = diffused_fa[..., t_idx_list, :, :, :]

        # print(f'Time to combine diffused quantities: {time.time()-tick:.3f}s')

        return diffused_fa, noised_seq, noised_seq_grad  #B,L,K

    @property
    def motif_com(self):
        """Center of mass of motif (for compatibility)"""
        return getattr(self, '_motif_com', torch.zeros(3))
    
    @motif_com.setter
    def motif_com(self, value):
        """Set center of mass of motif"""
        self._motif_com = value