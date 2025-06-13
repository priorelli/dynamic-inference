import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
import matplotlib.animation as animation
from pylab import tight_layout
import time
import sys
import config as c
import utils


def record_video(log, width):
    plot_type = 0
    frame = c.n_steps - 1
    dynamics = True

    # Initialize body
    idxs = {}
    ids = {joint: j for j, joint in enumerate(c.joints)}
    size = np.zeros((c.n_joints, 2))
    for joint in c.joints:
        size[ids[joint]] = c.joints[joint]['size']
        if c.joints[joint]['link']:
            idxs[ids[joint]] = ids[c.joints[joint]['link']]
        else:
            idxs[ids[joint]] = -1

    # Load variables
    angles = log['angles']
    pos = log['pos']
    est_pos = log['est_pos']

    ball_pos = log['ball_pos']
    est_ball_pos = log['est_ball_pos']

    square_pos = log['square_pos']
    est_square_pos = log['est_square_pos']

    true_vel = log['true_vel']
    est_vel = log['est_vel']
    F_m = log['F_m']

    L_int = log['L_int']

    L_softmax = np.zeros((len(L_int), 2))
    for s, step in enumerate(L_int):
        L_softmax[s] = utils.softmax(step * c.gain_evidence, c.w_bmc)

    # Create plot
    scale = 1.4
    x_range = (-c.width / 3, c.width / 3)
    y_range = (-c.height / 3, c.height / 3)
    if dynamics:
        fig = plt.figure(figsize=(40, (y_range[1] - y_range[0]) * 20 /
                                  (x_range[1] - x_range[0])))
        gs = GridSpec(1, 2, figure=fig)

        axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    else:
        fig = plt.figure(figsize=(
            20, (y_range[1] - y_range[0]) * 20 / (x_range[1] - x_range[0])))
        gs = GridSpec(1, 1, figure=fig)

        axs = [fig.add_subplot(gs[:, 0])]

    xlims = [x_range, (0, c.n_steps)]
    ylims = [y_range, (-0.05, 1.05)]

    def animate(n):
        if (n + 1) % 10 == 0:
            sys.stdout.write('\rStep: {:4d}'.format(n + 1))
            sys.stdout.flush()

        # Clear plot
        for w, xlim, ylim in zip(range(len(axs)), xlims, ylims):
            axs[w].clear()
            axs[w].set_xlim(xlim)
            axs[w].set_ylim(ylim)
        axs[0].get_xaxis().set_visible(False)
        axs[0].get_yaxis().set_visible(False)
        axs[1].set_ylabel('Prob.')
        axs[1].set_xlabel('Time')
        tight_layout()

        # Draw text
        axs[0].text(x_range[0] + 50, y_range[0] + 50, '%d' % (n % c.n_steps),
                    color='grey', size=50, weight='bold')

        for j in range(c.n_joints):
            # Draw real body
            axs[0].plot(*np.array([pos[n, idxs[j] + 1], pos[n, j + 1]]).T,
                        lw=size[j, 1] * scale, color='b', zorder=1)

        # Draw real ball
        ball_size = c.ball_size * scale * 100
        axs[0].scatter(*ball_pos[n], color='r', s=ball_size, zorder=0)

        # Draw real square
        rect = patches.Rectangle(
            square_pos[n] - [c.square_size / 2, c.square_size / 2],
            c.square_size / 2, c.square_size / 2,
            color='g', zorder=0)
        axs[0].add_patch(rect)

        # Draw estimated square
        # rect2 = patches.Rectangle(
        #     est_square_pos[n] - [c.square_size / 2, c.square_size / 2],
        #     c.square_size / 2, c.square_size / 2,
        #     color='olive', zorder=0)
        # axs[0].add_patch(rect2)

        # Draw estimated ball
        # axs[0].scatter(*est_ball_pos[n], color='purple', s=ball_size, zorder=0)

        # Draw real body trajectory
        # axs[0].scatter(*pos[n - (n % c.n_steps): n + 1, -1].T,
        #                color='darkblue', zorder=2)

        # Draw real ball trajectory
        # axs[0].scatter(*ball_pos[n - (n % c.n_steps): n + 1].T,
        #                color='darkred', zorder=2)

        # Draw real square trajectory
        # axs[0].scatter(*square_pos[n - (n % c.n_steps): n + 1].T,
        #                color='darkgreen', zorder=2)

        tg_vel = np.array([-np.sin(np.radians(angles[n, 0])),
                           np.cos(np.radians(angles[n, 0]))])
        # Draw quivers
        x_true, u_true = pos[n, -1], true_vel[n]
        x_pred1, u_pred1 = pos[n, -1], F_m[n, 0] * tg_vel
        x_pred2, u_pred2 = pos[n, -1], F_m[n, 1] * tg_vel

        q = axs[0].quiver(*x_true.T, *u_true.T, angles='xy', color='navy',
                          width=0.006, scale=400, alpha=0.8)
        q = axs[0].quiver(*x_pred1.T, *u_pred1.T, angles='xy',
                          color='r', width=0.006, scale=600)
        q = axs[0].quiver(*x_pred2.T, *u_pred2.T, angles='xy',
                          color='g', width=0.006, scale=600)

        if dynamics:
            axs[1].plot(np.repeat(L_softmax[:n + 1, 0][::c.n_tau], c.n_tau),
                        lw=width, label=r'$r_{t1}$', color='r')
            axs[1].plot(np.repeat(L_softmax[:n + 1, 1][::c.n_tau], c.n_tau),
                        lw=width, label=r'$r_{t2}$', color='g')

            axs[1].axhline(y=0, xmin=0, xmax=c.n_steps, color='black',
                           lw=1, zorder=0)

            axs[1].text(30, 0.96, 'Discrete outcomes',
                        color='grey', size=50, weight='bold')

    # Plot video
    if plot_type == 0:
        start = time.time()
        ani = animation.FuncAnimation(fig, animate, c.n_steps)
        writer = animation.writers['ffmpeg'](fps=60)
        ani.save('plots/video.mp4', writer=writer)
        print('\nTime elapsed:', time.time() - start)

    # Plot frame sequence
    elif plot_type == 1:
        for i in range(0, c.n_steps, c.n_steps // 10):
            animate(i)
            plt.savefig('plots/frame_%d' % i, bbox_inches='tight')

    # Plot single frame
    elif plot_type == 2:
        animate(frame)
        plt.savefig('plots/frame_%d' % frame, bbox_inches='tight')
