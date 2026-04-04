import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


class LQRSolver:
    def __init__(self, H, M, C, D, R, sigma, T):
        """
        H, M, C, D, R, sigma: 2x2 matrices
        T: terminal time
        """
        self.H = np.array(H, dtype=float)
        self.M = np.array(M, dtype=float)
        self.C = np.array(C, dtype=float)
        self.D = np.array(D, dtype=float)
        self.R = np.array(R, dtype=float)
        self.sigma = np.array(sigma, dtype=float)
        self.T = float(T)

        self.D_inv = np.linalg.inv(self.D)

        self.time_grid = None
        self.S_grid = None
        self.trace_integral_grid = None

        self.S_interp = None
        self.trace_interp = None

    def _riccati_rhs(self, t, s_flat):
        """
        Riccati ODE:
        S'(t) = -2 H^T S + S M D^{-1} M^T S - C
        """
        S = s_flat.reshape(2, 2)

        rhs = (
            -2.0 * self.H.T @ S
            + S @ self.M @ self.D_inv @ self.M.T @ S
            - self.C
        )

        return rhs.reshape(-1)

    def solve_riccati(self, time_grid):
        """
        Solve Riccati ODE backward from T to 0 on the given time grid.
        time_grid: 1D numpy array or torch tensor, increasing order
        """
        if isinstance(time_grid, torch.Tensor):
            time_grid = time_grid.detach().cpu().numpy()

        time_grid = np.array(time_grid, dtype=float)

        assert time_grid.ndim == 1, "time_grid must be 1D"
        assert np.all(np.diff(time_grid) > 0), "time_grid must be strictly increasing"
        assert time_grid[0] >= 0.0 and time_grid[-1] <= self.T + 1e-12

        sol = solve_ivp(
            fun=self._riccati_rhs,
            t_span=(self.T, 0.0),
            y0=self.R.reshape(-1),
            t_eval=time_grid[::-1],
            rtol=1e-9,
            atol=1e-11
        )

        if not sol.success:
            raise RuntimeError("Riccati ODE solver failed.")

        S_rev = sol.y.T.reshape(-1, 2, 2)
        S_grid = S_rev[::-1]

        self.time_grid = time_grid
        self.S_grid = S_grid

        # integral term: int_t^T tr(sigma sigma^T S(r)) dr
        A = self.sigma @ self.sigma.T
        tr_vals = np.array([np.trace(A @ S) for S in S_grid])

        integral = np.zeros_like(tr_vals)
        for i in range(len(time_grid) - 2, -1, -1):
            dt = time_grid[i + 1] - time_grid[i]
            integral[i] = integral[i + 1] + 0.5 * dt * (tr_vals[i] + tr_vals[i + 1])

        self.trace_integral_grid = integral

        self.S_interp = interp1d(
            self.time_grid,
            self.S_grid,
            axis=0,
            kind="linear",
            fill_value="extrapolate"
        )

        self.trace_interp = interp1d(
            self.time_grid,
            self.trace_integral_grid,
            kind="linear",
            fill_value="extrapolate"
        )

    def get_S(self, t_batch):
        if self.S_interp is None:
            raise RuntimeError("Please call solve_riccati(time_grid) first.")

        if isinstance(t_batch, torch.Tensor):
            t_np = t_batch.detach().cpu().numpy()
        else:
            t_np = np.array(t_batch, dtype=float)

        S_np = self.S_interp(t_np)
        return torch.tensor(S_np, dtype=torch.float32)

    def get_trace_integral(self, t_batch):
        if self.trace_interp is None:
            raise RuntimeError("Please call solve_riccati(time_grid) first.")

        if isinstance(t_batch, torch.Tensor):
            t_np = t_batch.detach().cpu().numpy()
        else:
            t_np = np.array(t_batch, dtype=float)

        val_np = self.trace_interp(t_np)
        return torch.tensor(val_np, dtype=torch.float32)

    def value_function(self, t_batch, x_batch):
        """
        t_batch: torch tensor of shape (batch,)
        x_batch: torch tensor of shape (batch, 1, 2)

        returns: torch tensor of shape (batch, 1)
        """
        S_batch = self.get_S(t_batch)
        integral_batch = self.get_trace_integral(t_batch)

        x = x_batch.squeeze(1).float()
        S_batch = S_batch.to(x.device)
        integral_batch = integral_batch.to(x.device)

        x_col = x.unsqueeze(-1)
        x_row = x.unsqueeze(1)

        quad = torch.bmm(torch.bmm(x_row, S_batch), x_col).squeeze(-1)
        val = quad + integral_batch.unsqueeze(-1)
        return val

    def optimal_control(self, t_batch, x_batch):
        """
        t_batch: torch tensor of shape (batch,)
        x_batch: torch tensor of shape (batch, 1, 2)

        returns: torch tensor of shape (batch, 2)
        """
        S_batch = self.get_S(t_batch)

        x = x_batch.squeeze(1).float()
        S_batch = S_batch.to(x.device)

        D_inv = torch.tensor(self.D_inv, dtype=torch.float32, device=x.device)
        M_T = torch.tensor(self.M.T, dtype=torch.float32, device=x.device)

        x_col = x.unsqueeze(-1)
        Sx = torch.bmm(S_batch, x_col)

        A = -torch.matmul(D_inv, M_T)
        a = torch.matmul(A.unsqueeze(0), Sx).squeeze(-1)
        return a
