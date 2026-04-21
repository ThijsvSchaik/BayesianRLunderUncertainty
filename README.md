# BayesianRLunderUncertainty

Minimal starter implementation for a 2D reinforcement-learning environment with:

- A 100x100 state space `(i, j)`
- Actions `Left`, `Right`, `Up`, `Down`
- Stochastic transitions per action
- A Gaussian reward surface `R(s) = X(i,j) * phi`, where `phi ~ N(1, 0.1)`
- Unsafe states with state-dependent falling probability

## Quick start

```python
from bayesian_rl_under_uncertainty import GridGaussianEnvironment

env = GridGaussianEnvironment(seed=42)
state = env.reset()
next_state, reward, done, info = env.step("Right")
```

## Visualizations

```python
env.plot_reward_map()
env.plot_unsafe_states()
```

## Notebook

See `notebooks/environment_quickstart.ipynb` for a simple experiment flow.
