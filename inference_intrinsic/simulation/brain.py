import torch
import numpy as np
import utils
import config as c
from simulation.unit import Unit, Obs
from simulation.discrete import Discrete


# Get proprioceptive prediction
def g_prop(x):
    return x[0, 0]


# Get visual prediction
def g_vis(x):
    length_norm = utils.normalize(c.lengths, c.norm_cart)[0]

    return utils.kinematics(x, length_norm, c.norm_polar)


# Stay
def f_0(x, lmbda):
    return x * 0.0


# Reach ball
def f_b(x, lmbda):
    return (torch.stack([x[1], x[1], x[2]], -2) - x) * lmbda


# Reach square
def f_s(x, lmbda):
    return (torch.stack([x[2], x[1], x[2]], -2) - x) * lmbda


# Define brain class
class Brain:
    def __init__(self):
        # Initialize discrete
        self.discrete = Discrete()

        # Initialize units
        self.int = Unit(dim=(c.n_orders, c.n_objects, c.n_joints),
                        inputs=utils.normalize(c.eta_x_int, c.norm_polar),
                        v=self.discrete.o_int, L=self.discrete.L_int,
                        pi_eta_x=c.pi_eta_x_int, p_x=c.p_x_int,
                        pi_x=c.pi_x_int, lr=c.lr_int,
                        F_m=[f_b, f_s], lmbda=c.lambda_int)

        self.prop = Obs(dim=c.n_joints, inputs=[self.int],
                        pi_o=c.pi_prop, g=g_prop, lr=c.lr_a)

        self.vis = Obs(dim=(c.n_orders, c.n_objects, 2), inputs=[self.int],
                       pi_o=c.pi_vis, g=g_vis)

        self.units = [self.int, self.prop, self.vis]

    # Initialize beliefs
    def init_belief(self, angles, pos):
        int_start = angles if c.x_int_start is None else c.x_int_start
        int_start_norm = utils.normalize(int_start, c.norm_polar)
        self.int.x[0, :] = torch.tensor(int_start_norm)

    # Run an inference step
    def inference_step(self, O, step):
        # Run discrete step
        if (step + 1) % c.n_tau == 0:
            self.discrete.step()

        # Set observations
        self.prop.o = torch.tensor(O[0])
        self.vis.o = torch.tensor(O[1])

        # Perform message passing step
        for unit in self.units:
            unit.step()

        # Update all units
        for unit in self.units:
            unit.update(c.dt)

        return utils.denormalize(self.prop.actions, c.norm_polar)
