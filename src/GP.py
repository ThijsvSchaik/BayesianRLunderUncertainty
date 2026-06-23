from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
import torch
import gpytorch

class BoTorchDualSingleTaskGP:
    def __init__(self, train_x, train_y, normalize_inputs=True, standardize_outputs=True, 
                 x_kernel=gpytorch.kernels.RBFKernel, y_kernel=gpytorch.kernels.RBFKernel):
        train_x = train_x.to(torch.float64)
        train_y = train_y.to(torch.float64)
        
        self.num_tasks = 2
        n_inputs = train_x.shape[-1]
        self.train_x = train_x
        self.train_y = train_y

        #x_pos (0-100), y_pos (0-100), action_x (-1 to 1), action_y (-1 to 1)
        bounds = torch.tensor([
            [0., 0., -1., -1.],
            [100., 100., 1., 1.]
        ], dtype=torch.float64)
        
        # Two independent SingleTaskGPs with built-in normalization
        self.gp_x = SingleTaskGP(
            train_X=train_x,
            train_Y=train_y[:, 0:1],
            input_transform=Normalize(d=n_inputs, bounds=bounds) if normalize_inputs else None,
            outcome_transform=Standardize(m=1) if standardize_outputs else None,
            covar_module=x_kernel()
        )
        
        self.gp_y = SingleTaskGP(
            train_X=train_x,
            train_Y=train_y[:, 1:2],
            input_transform=Normalize(d=n_inputs, bounds=bounds) if normalize_inputs else None,
            outcome_transform=Standardize(m=1) if standardize_outputs else None,
            covar_module=y_kernel()
        )
        
        self.mll_x = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp_x.likelihood, self.gp_x)
        self.mll_y = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp_y.likelihood, self.gp_y)
    
    def fit(self, training_iterations=50, lr=0.1, verbose=True):
        self.gp_x.train()
        self.gp_y.train()
        
        optimizer_x = torch.optim.Adam(self.gp_x.parameters(), lr=lr)
        optimizer_y = torch.optim.Adam(self.gp_y.parameters(), lr=lr)
        
        losses_x = []
        losses_y = []
        
        for i in range(training_iterations):
            # Train x GP
            optimizer_x.zero_grad()
            output_x = self.gp_x(self.train_x)
            loss_x = -self.mll_x(output_x, self.train_y[:, 0])
            loss_x.backward()
            optimizer_x.step()
            losses_x.append(loss_x.item())
            
            # Train y GP
            optimizer_y.zero_grad()
            output_y = self.gp_y(self.train_x)
            loss_y = -self.mll_y(output_y, self.train_y[:, 1])
            loss_y.backward()
            optimizer_y.step()
            losses_y.append(loss_y.item())
            
            if verbose and (i + 1) % 10 == 0:
                print(f'Iter {i+1}/{training_iterations} - Loss X: {loss_x.item():.3f}, Loss Y: {loss_y.item():.3f}')
        
        return losses_x, losses_y
    
    def predict(self, x):
        """
        Get prediction at x. Returns mean and covariance in raw space.
        Only combines outputs here at prediction time.
        """
        x = x.to(torch.float64)
        
        self.gp_x.eval()
        self.gp_y.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Get posterior from each GP (already untransformed by BoTorch)
            posterior_x = self.gp_x.posterior(x)
            posterior_y = self.gp_y.posterior(x)
            
            # Means: (n, 1) each -> stack to (n, 2)
            mean_x = posterior_x.mean.squeeze(-1)  # (n,)
            mean_y = posterior_y.mean.squeeze(-1)  # (n,)
            mean = torch.stack([mean_x, mean_y], dim=-1)  # (n, 2)
            
            # Variances: (n, 1, 1) each -> extract diagonal
            var_x = posterior_x.variance.squeeze(-1)  # (n,)
            var_y = posterior_y.variance.squeeze(-1)  # (n,)
            
            # Build 2x2 covariance per point (independent outputs -> diagonal)
            # For single point: just return 2x2 diagonal covariance
            if x.shape[0] == 1:
                cov = torch.diag(torch.stack([var_x.squeeze(), var_y.squeeze()]))
            else:
                # For multiple points, return per-point 2x2 covariances
                cov = torch.zeros(x.shape[0], 2, 2, dtype=x.dtype)
                cov[:, 0, 0] = var_x
                cov[:, 1, 1] = var_y
        
        return mean, cov
    
    def eval(self):
        """Set both GPs to eval mode."""
        self.gp_x.eval()
        self.gp_y.eval()
        return self
    
    def get_random_train_data(env, 
                                actions = {
                                    'up': (0, -1),
                                    'right': (1, 0),
                                    'down': (0, 1),
                                    'left': (-1, 0),
                                }, 
                                start_positions = [(50, 50), (20, 20), (80, 80), (20, 80), (80, 20)],
                                walk_length=500,
                                walks=1
                                ):
        # Generate training data
        import random

        actions_taken = [random.choice(list(actions.values())) for _ in range(walk_length)]
        state = start_positions[0]  # Start at the first position for the first walk

        # N samples x M tasks, (0,4) as we have not sampled yet
        train_x = torch.empty((0, 4), dtype=torch.float64)
        train_y = torch.empty((0, 2), dtype=torch.float64)

        states_visited = []
        unsafe_count = 0

        for start_pos in start_positions:
            state = start_pos
            for walk in range(walks):
                for i in range(walk_length):
                    x, y = state
                    states_visited.append(state)
                    state = env.transition(state=state, action=actions_taken[i])
                    train_x = torch.cat((train_x, torch.tensor([[x, y, actions_taken[i][0], actions_taken[i][1]]], dtype=torch.float64)), dim=0)
                    train_y = torch.cat((train_y, torch.tensor([[state[0], state[1]]], dtype=torch.float64)), dim=0)
                    if env.unsafe_mask[state]:
                        unsafe_count += 1  # Stop walk if we hit an unsafe state


        # for i in range(walk_length):
        #     x, y = state
        #     state = env.transition(state=start_state, action=actions_taken[i])
        #     train_x = torch.cat((train_x, torch.tensor([[start_state[0], start_state[1], actions_taken[i][0], actions_taken[i][1]]])), dim=0)
        #     train_y = torch.cat((train_y, torch.tensor([[state[0], state[1]]])), dim=0)

        return train_x, train_y, {x: states_visited.count(x) for x in states_visited}, unsafe_count