import torch
from torch.optim import Optimizer

"""
Code based on https://github.com/thetechdude124/Adam-Optimization-From-Scratch/blob/master/CustomAdam.py
"""


class OrthogonalAdam(Optimizer):
    
    """
    A custom implementation of the Adam optimizer. Defaults used are as recommended in https://arxiv.org/abs/1412.6980 
    See the paper or visit Optimizer_Experimentation.ipynb for more information on how exactly Adam works + mathematics behind it.

    Params:
    lr (float): the effective upperbound of the optimizer step in most cases (size of step). DEFAULT - 0.001.
    betas (float): bias for the first and second moment estimate. DEFAULT - (0.9, 0.999).
    eps (float): small number added to prevent division by zero, DEFAULT - 10e-8.
    bias_correction (bool): whether the optimizer should correct for the specified biases when taking a step. DEFAULT - TRUE.
    """
    # Initialize optimizer with parameters
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.):
        # Check if lr and biases are invalid (negative)
        if lr < 0:
            raise ValueError("Invalid lr [{}]. Choose a positive lr".format(lr))
        if betas[0] < 0 or betas[1] < 0:
            raise ValueError("Invalid bias parameters [{}, {}]. Choose positive bias parameters.".format(betas[0], betas[1]))
        # Declare dictionary of default values for optimizer initialization
        DEFAULTS = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # Initialize the optimizer
        super(OrthogonalAdam, self).__init__(params, DEFAULTS)

    # Step method (for updating parameters)
    def step(self, closure = None):
        # Set loss to none
        loss = None
        # If the closure is set to True, set the loss to the closure function
        loss = closure() if closure != None else loss
        # Check if this is the first step - if not, increment the current step
        if not self.state["step"]:
            self.state["step"] = 1
            self.state["first_moment_estimate"] = []
            self.state["second_moment_estimate"] = []
            # Use Adam optimization method - first, define all the required arguments for the parameter if we are on the first step
            for i, param_group in enumerate(self.param_groups):
                self.state["first_moment_estimate"].append([])
                self.state["second_moment_estimate"].append([])
                for param in param_group["params"]:
                    self.state["first_moment_estimate"][i].append(torch.zeros_like(param.data))
                    self.state["second_moment_estimate"][i].append(torch.zeros_like(param.data))
        else:
            self.state["step"] += 1
        # Iterate over "groups" of parameters (layers of parameters in the network) to begin processing and computing the next set of params
        for i, param_group in enumerate(self.param_groups):
            # Iterate over individual parameters
            for j, param in enumerate(param_group["params"]):
                # Check if gradients have been computed for each parameter
                # If not - if there are no gradients - then skip the parameter
                if param.grad is None:
                    continue
                gradients = param.grad.data
                # Declare variables from state - inplace methods modify state variable directly
                first_moment_estimate = self.state["first_moment_estimate"][i][j]
                second_moment_estimate = self.state["second_moment_estimate"][i][j]
                # Project out previous timestep gradient direction
                proj = (gradients*first_moment_estimate).sum()/(first_moment_estimate.pow(2).sum() + param_group["eps"]) * first_moment_estimate
                gradients.add_(-proj)
                # Compute the first moment estimate - B_1 * m_t + (1-B_1) * grad (uncentered)
                first_moment_estimate.mul_(param_group["betas"][0]).add_(gradients * (1.0 - param_group["betas"][0]))
                # Compute the second moment estimate - B_2 * v_t + (1-B_2) * grad^2 (uncentered)
                second_moment_estimate.mul_(param_group["betas"][1]).add_(gradients.pow(2) * (1.0 - param_group["betas"][1]))
                # Perform bias correction for the first moment estimate: m_t / (1 -B_1^t)
                first_moment_estimate_hat = first_moment_estimate.divide(1.0 - (param_group["betas"][0] ** self.state["step"]))
                # Perform bias correction for second moment estimate: v_t / (1 - B_2^t)
                second_moment_estimate_hat = second_moment_estimate.divide(1.0 - (param_group["betas"][1] ** self.state["step"]))
                # Apply weight decay
                if param_group["weight_decay"] > 0.:
                    # Add the weight decay to the parameter
                    param.data.add_(-param_group["lr"] * param_group["weight_decay"] * param.data)
                # Next, perform the actual update
                param.data.add_((-param_group["lr"]) * first_moment_estimate_hat.divide(second_moment_estimate_hat.sqrt() + param_group["eps"]))
        # Return the loss
        return loss
