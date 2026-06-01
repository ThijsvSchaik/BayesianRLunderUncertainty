from typing import List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

@dataclass
class ProbabilisticSimpleSystem:
    def __init__(self, seed, 
                hill_importance=1.0,
                grid_shape=(100, 100),
                reward_gaussian_mean=(80, 65),
                reward_gaussian_sigma=20.0,
                reward_scale=1.0,
                num_unsafe_blocks=10,
                block_size_range=(6, 18),
                step_size=3.0,
                sigma_parallel=1.5,
                sigma_perpendicular=0.5

            ):
        
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.grid_shape = grid_shape
        self.reward_gaussian_mean = reward_gaussian_mean
        self.reward_gaussian_sigma = reward_gaussian_sigma
        self.reward_scale = reward_scale
        self.num_unsafe_blocks = num_unsafe_blocks
        self.block_size_range = block_size_range
        self.step_size = step_size
        self.sigma_parallel = sigma_parallel
        self.sigma_perpendicular = sigma_perpendicular

        self.reward_map = self._build_reward_map()
        self.unsafe_mask = self._build_unsafe_mask()
        self.hill_importance = hill_importance
        
        rows, cols = self.grid_shape
        self.hill_vx = np.zeros((rows, cols))
        self.hill_vy = np.zeros((rows, cols))
        self.hill_tops = []  # List of (top, sigma, strength, normalized)

    def _build_unsafe_mask(self):
        rows, cols = self.grid_shape
        mask = np.zeros((rows, cols), dtype=bool)

        for _ in range(self.num_unsafe_blocks):
            w = self.rng.integers(self.block_size_range[0], self.block_size_range[1] + 1)
            h = self.rng.integers(self.block_size_range[0], self.block_size_range[1] + 1)
            x0 = self.rng.integers(0, self.grid_shape[1] - w)
            y0 = self.rng.integers(0, self.grid_shape[0] - h)
            mask[y0:y0 + h, x0:x0 + w] = True

        # Make sure starting position is always safe
        mask[0][0] = False
        return mask

    def _build_falling_prob_map(self):
        # todo
        pass

    def transition(self, state: Tuple[int, int], action: Tuple[int, int]) -> Tuple[int, int]:
        x, y = state
        hill_influence_x = self.hill_vx[y, x] * self.hill_importance
        hill_influence_y = self.hill_vy[y, x] * self.hill_importance
        
        # print(f"Hill influence at state {state}: (x={hill_influence_x:.2f}, y={hill_influence_y:.2f})")
        
        dx, dy = action
        
        mean_x = x + self.step_size * dx
        mean_y = y + self.step_size * dy
        
        if dx != 0:  # horizontal movement (right/left)
            cov = np.array([
                [self.sigma_parallel**2, 0],
                [0, self.sigma_perpendicular**2]
            ])
        else:  # vertical movement (up/down)
            cov = np.array([
                [self.sigma_perpendicular**2, 0],
                [0, self.sigma_parallel**2]
            ])
        
        new_pos = self.rng.multivariate_normal([mean_x, mean_y], cov)
        
        # Make sure they stay within bounds and apply hill influence
        new_x = int(np.clip(np.round(new_pos[0] +hill_influence_x), 0, self.grid_shape[0] - 1)) 
        new_y = int(np.clip(np.round(new_pos[1]+ hill_influence_y), 0, self.grid_shape[1] - 1))
        
        return (new_x, new_y)

    # Reward without noise
    def _build_reward_map(self):
        rows, cols = self.grid_shape
        mean_i, mean_j = self.reward_gaussian_mean
        sigma2 = self.reward_gaussian_sigma ** 2

        reward_map = [
            [
                float(np.exp(-((i - mean_i) ** 2 + (j - mean_j) ** 2) / (2.0 * sigma2)))
                for j in range(cols)
            ]
            for i in range(rows)
        ]
        return reward_map
    
    # Reward per state
    def get_reward(self, state):
        base_reward = self.reward_map[state[0]][state[1]]
        return base_reward * self.reward_scale * self.rng.normal(1, 0.1)

    def _generate_hill(self, top, sigma=20.0):
        cx, cy = top
        rows, cols = self.grid_shape
        
        x = np.arange(cols)
        y = np.arange(rows)
        X, Y = np.meshgrid(x, y)
        
        # Gaussian hill centered at (cx, cy)
        Z = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
        gy, gx = np.gradient(Z)
        
        vx = -gx
        vy = gy

        return (X, Y, vx, vy, Z)

    def add_hill(self, top: Tuple[int, int], sigma: float = 20.0, strength: float = 1.0, normalized: bool = False):
        X, Y, vx, vy, Z = self._generate_hill(top, sigma)
        
        if normalized:
            mag = np.sqrt(vx**2 + vy**2) + 1e-8
            vx = vx / mag
            vy = vy / mag
        
        self.hill_vx += strength * vx
        self.hill_vy += strength * vy
        self.hill_tops.append((top, sigma, strength, normalized))

    def clear_hills(self):
        rows, cols = self.grid_shape
        self.hill_vx = np.zeros((rows, cols))
        self.hill_vy = np.zeros((rows, cols))
        self.hill_tops = []

    def plot_reward_map(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        data = np.asarray(self.reward_map, dtype=float)
        if data.ndim != 2:
            raise ValueError(f"reward_map must be 2D, got shape {data.shape}")

        rows, cols = data.shape
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        im = ax.imshow(
            data,
            cmap="viridis",
            origin="upper",
            interpolation="nearest",
            vmin=np.min(data),
            vmax=np.max(data),
        )

        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.15, alpha=0.3)
        ax.tick_params(which="minor", bottom=False, left=False)

        step = 10
        ax.set_xticks(np.arange(0, cols, step))
        ax.set_yticks(np.arange(0, rows, step))

        ax.set_title("Reward Map (PSS)")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.figure.colorbar(im, ax=ax, label="Reward")
        return ax

    def plot_unsafe_mask(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        data = np.asarray(self.unsafe_mask, dtype=bool)
        if data.ndim != 2:
            raise ValueError(f"unsafe_mask must be 2D, got shape {data.shape}")

        rows, cols = data.shape
        cmap = ListedColormap(["#ffffff", "#d62728"])
        norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        im = ax.imshow(
            data.astype(int),
            cmap=cmap,
            norm=norm,
            origin="upper",
            interpolation="nearest",
        )

        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=0.15, alpha=0.25)
        ax.tick_params(which="minor", bottom=False, left=False)

        step = 10
        ax.set_xticks(np.arange(0, cols, step))
        ax.set_yticks(np.arange(0, rows, step))

        ax.set_title("Unsafe States (PSS)")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        cbar = ax.figure.colorbar(im, ax=ax, ticks=[0, 1])
        cbar.ax.set_yticklabels(["Safe", "Unsafe"])
        cbar.set_label("State Type")
        return ax

    def plot_combined(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        reward_data = np.asarray(self.reward_map, dtype=float)
        unsafe_data = np.asarray(self.unsafe_mask, dtype=bool)

        rows, cols = reward_data.shape

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        im = ax.imshow(
            reward_data,
            cmap="viridis",
            origin="upper",
            interpolation="nearest",
            vmin=np.min(reward_data),
            vmax=np.max(reward_data),
        )
        ax.figure.colorbar(im, ax=ax, label="Reward", fraction=0.046, pad=0.04)

        unsafe_rgba = np.zeros((rows, cols, 4), dtype=float)
        unsafe_rgba[unsafe_data] = [0.85, 0.15, 0.15, 0.45]
        ax.imshow(unsafe_rgba, origin="upper", interpolation="nearest")

        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.15, alpha=0.2)
        ax.tick_params(which="minor", bottom=False, left=False)

        step = 10
        ax.set_xticks(np.arange(0, cols, step))
        ax.set_yticks(np.arange(0, rows, step))

        ax.set_title("Reward Map with Unsafe States Overlay")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.legend(
            handles=[Patch(facecolor=(0.85, 0.15, 0.15, 0.45), label="Unsafe")],
            loc="upper left",
        )
        return ax
    

    def plot_hills(self, color='black', ax: Optional[plt.Axes] = None, x=None, y=None) -> plt.Axes:
        rows, cols = self.grid_shape
        
        if x is None:
            x = np.arange(cols)
        if y is None:
            y = np.arange(rows)

        X, Y = np.meshgrid(x, y)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        skip = max(1, rows // 15)

        ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                  self.hill_vx[::skip, ::skip], self.hill_vy[::skip, ::skip],
                  color=color, alpha=0.8, scale=25)
        
        for i, (top, sigma, strength, normalized) in enumerate(self.hill_tops):
            tx, ty = top
            ax.plot(tx + 0.5, ty + 0.5, 'o', markersize=5, 
                    label=f'Hill {i+1}' if len(self.hill_tops) > 1 else 'Hill top')
        
        ax.set_xlim(0, cols)
        ax.set_ylim(rows, 0)  # Flip y to match imshow

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(loc='upper right')
        return ax