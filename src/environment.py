from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, isclose
from random import Random
from typing import Dict, List, Optional, Tuple

State = Tuple[int, int]
TransitionModel = Dict[str, Dict[str, float]]
FloatGrid = List[List[float]]
BoolGrid = List[List[bool]]


@dataclass
class GridGaussianEnvironment:
    """2D grid environment with Gaussian rewards and unsafe falling states."""

    grid_shape: Tuple[int, int] = (100, 100)
    start_state: State = (0, 0)
    unsafe_fraction: float = 0.1
    unsafe_fall_prob_range: Tuple[float, float] = (0.05, 0.3)
    gaussian_sigma: float = 15.0
    transition_model: Optional[TransitionModel] = None
    unsafe_mask: Optional[BoolGrid] = None
    falling_prob_map: Optional[FloatGrid] = None
    seed: Optional[int] = None

    actions: Tuple[str, ...] = ("Left", "Right", "Up", "Down")
    _action_deltas: Dict[str, Tuple[int, int]] = field(
        default_factory=lambda: {
            "Left": (0, -1),
            "Right": (0, 1),
            "Up": (-1, 0),
            "Down": (1, 0),
        }
    )

    def __post_init__(self) -> None:
        self.rng = Random(self.seed)
        self.transition_model = self.transition_model or {
            "Left": {"Left": 0.8, "Up": 0.1, "Down": 0.1},
            "Right": {"Right": 0.8, "Up": 0.1, "Down": 0.1},
            "Up": {"Up": 0.8, "Left": 0.1, "Right": 0.1},
            "Down": {"Down": 0.8, "Left": 0.1, "Right": 0.1},
        }
        self._validate_transition_model()

        self.reward_map, self.reward_mean = self._build_reward_map()
        self.unsafe_mask = self._build_unsafe_mask(self.unsafe_mask)
        self.falling_prob_map = self._build_falling_prob_map(self.falling_prob_map)

        self.current_state = self.start_state
        self.terminated = False

    def _validate_transition_model(self) -> None:
        for action in self.actions:
            if action not in self.transition_model:
                raise ValueError(f"Missing transition probabilities for action '{action}'")
            total = sum(self.transition_model[action].values())
            if not isclose(total, 1.0, rel_tol=1e-9):
                raise ValueError(f"Transition probabilities for '{action}' must sum to 1, got {total}")

    def _build_reward_map(self) -> Tuple[FloatGrid, Tuple[float, float]]:
        rows, cols = self.grid_shape
        mean_i = self.rng.uniform(0, rows - 1)
        mean_j = self.rng.uniform(0, cols - 1)
        denom = 2.0 * self.gaussian_sigma**2

        reward_map: FloatGrid = []
        for i in range(rows):
            row: List[float] = []
            for j in range(cols):
                dist2 = (i - mean_i) ** 2 + (j - mean_j) ** 2
                row.append(exp(-dist2 / denom))
            reward_map.append(row)
        return reward_map, (mean_i, mean_j)

    def _build_unsafe_mask(self, unsafe_mask: Optional[BoolGrid]) -> BoolGrid:
        rows, cols = self.grid_shape
        if unsafe_mask is not None:
            self._validate_grid_shape(unsafe_mask, expected=self.grid_shape)
            mask = [[bool(v) for v in row] for row in unsafe_mask]
            mask[self.start_state[0]][self.start_state[1]] = False
            return mask

        mask = [
            [self.rng.random() < self.unsafe_fraction for _ in range(cols)]
            for _ in range(rows)
        ]
        mask[self.start_state[0]][self.start_state[1]] = False
        return mask

    def _build_falling_prob_map(self, falling_prob_map: Optional[FloatGrid]) -> FloatGrid:
        rows, cols = self.grid_shape
        if falling_prob_map is not None:
            self._validate_grid_shape(falling_prob_map, expected=self.grid_shape)
            prob_map: FloatGrid = []
            for i in range(rows):
                row: List[float] = []
                for j in range(cols):
                    value = float(falling_prob_map[i][j])
                    if value < 0.0 or value > 1.0:
                        raise ValueError("falling_prob_map values must be in [0, 1]")
                    row.append(value if self.unsafe_mask[i][j] else 0.0)
                prob_map.append(row)
            return prob_map

        low, high = self.unsafe_fall_prob_range
        prob_map = [[0.0 for _ in range(cols)] for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                if self.unsafe_mask[i][j]:
                    prob_map[i][j] = self.rng.uniform(low, high)
        return prob_map

    @staticmethod
    def _validate_grid_shape(grid, expected: Tuple[int, int]) -> None:
        rows, cols = expected
        if len(grid) != rows:
            raise ValueError(f"Grid must have {rows} rows")
        if any(len(row) != cols for row in grid):
            raise ValueError(f"Grid rows must all have {cols} columns")

    def reset(self) -> State:
        self.current_state = self.start_state
        self.terminated = False
        return self.current_state

    def transition_distribution(self, state: State, action: str) -> Dict[State, float]:
        if action not in self.actions:
            raise ValueError(f"Unknown action '{action}'")

        result: Dict[State, float] = {}
        for sampled_action, prob in self.transition_model[action].items():
            next_state = self._apply_action(state, sampled_action)
            result[next_state] = result.get(next_state, 0.0) + prob
        return result

    def _apply_action(self, state: State, action: str) -> State:
        di, dj = self._action_deltas[action]
        rows, cols = self.grid_shape
        next_i = max(0, min(rows - 1, state[0] + di))
        next_j = max(0, min(cols - 1, state[1] + dj))
        return next_i, next_j

    def sample_reward(self, state: State) -> float:
        phi = self.rng.gauss(1.0, 0.1)
        return float(self.reward_map[state[0]][state[1]] * phi)

    def step(self, action: str) -> Tuple[State, float, bool, Dict[str, bool]]:
        if self.terminated:
            raise RuntimeError("Environment is terminated. Call reset() to continue.")
        if action not in self.actions:
            raise ValueError(f"Unknown action '{action}'")

        transitions = self.transition_model[action]
        sampled = self.rng.random()
        cumulative = 0.0
        sampled_action = action
        for possible_action, prob in transitions.items():
            cumulative += prob
            if sampled <= cumulative:
                sampled_action = possible_action
                break

        self.current_state = self._apply_action(self.current_state, sampled_action)

        reward = self.sample_reward(self.current_state)
        i, j = self.current_state
        fell = self.unsafe_mask[i][j] and self.rng.random() < self.falling_prob_map[i][j]
        self.terminated = bool(fell)

        return self.current_state, reward, self.terminated, {"fell": bool(fell)}

    def plot_reward_map(self, ax=None):
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("matplotlib is required for plotting") from exc

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5))
        image = ax.imshow(self.reward_map, origin="lower", cmap="viridis")
        ax.set_title("Gaussian Reward Map")
        ax.set_xlabel("j")
        ax.set_ylabel("i")
        plt.colorbar(image, ax=ax, label="X(i,j)")
        return ax

    def plot_unsafe_states(self, ax=None):
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("matplotlib is required for plotting") from exc

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(self.unsafe_mask, origin="lower", cmap="Reds")
        ax.set_title("Unsafe States")
        ax.set_xlabel("j")
        ax.set_ylabel("i")
        return ax
