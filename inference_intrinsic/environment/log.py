import numpy as np
import config as c
import utils


# Define log class
class Log:
    def __init__(self):
        # Initialize logs
        self.angles = np.zeros((c.n_steps, c.n_joints))
        self.est_angles = np.zeros_like(self.angles)

        self.pos = np.zeros((c.n_steps, c.n_joints + 1, 2))
        self.est_pos = np.zeros_like(self.pos)

        self.ball_pos = np.zeros((c.n_steps, 2))
        self.est_ball_pos = np.zeros_like(self.ball_pos)

        self.square_pos = np.zeros((c.n_steps, 2))
        self.est_square_pos = np.zeros_like(self.square_pos)

        self.causes_int = np.zeros((c.n_steps, 2))
        self.true_vel = np.zeros((c.n_steps, 2))
        self.est_vel = np.zeros(c.n_steps)
        self.F_m = np.zeros((c.n_steps, 2))
        self.L_int = np.zeros((c.n_steps, 2))

    # Track logs for each iteration
    def track(self, step, brain, body, ball, square):
        self.angles[step] = body.get_angles()
        est_angles = brain.prop.predict().detach().numpy()
        self.est_angles[step] = utils.denormalize(est_angles, c.norm_polar)

        self.pos[step, 1:] = body.get_pos()
        self.est_pos[step] = body.get_poses(self.est_angles[step],
                                            c.lengths)[:, :2]

        self.ball_pos[step] = ball.get_pos()[-1]
        est_ball_pos = brain.vis.predict()[0, 1].detach().numpy()
        self.est_ball_pos[step] = utils.denormalize(est_ball_pos, c.norm_cart)

        self.square_pos[step] = square.get_pos()[-1]
        est_square_pos = brain.vis.predict()[0, 2].detach().numpy()
        self.est_square_pos[step] = utils.denormalize(est_square_pos,
                                                      c.norm_cart)

        self.causes_int[step] = brain.int.v

        self.true_vel[step] = body.joints[-1].get_vel()
        self.est_vel[step] = utils.denormalize(
            brain.int.x[1, 0][0].detach().numpy(), c.norm_polar)

        self.F_m[step] = utils.denormalize(
            brain.int.Preds_x[:, 0].T[0].detach().numpy(), c.norm_polar)
        self.L_int[step] = brain.discrete.L_int

    # Save log to file
    def save_log(self):
        np.savez_compressed('simulation/log_{}'.format(c.log_name),
                            angles=self.angles,
                            est_angles=self.est_angles,
                            pos=self.pos,
                            est_pos=self.est_pos,
                            ball_pos=self.ball_pos,
                            est_ball_pos=self.est_ball_pos,
                            square_pos=self.square_pos,
                            est_square_pos=self.est_square_pos,
                            causes_int=self.causes_int,
                            true_vel=self.true_vel,
                            est_vel=self.est_vel,
                            F_m=self.F_m,
                            L_int=self.L_int)
