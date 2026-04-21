from typing import Tuple

class ProbabilisticSimpleSystem:
    grid_shape: Tuple[int, int] = (100, 100)
    reward_gaussian_sigma: 10.0
    reward_gaussian_mean: Tuple[int, int] = (65, 80)
