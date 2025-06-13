# Window
width = 1000
height = 800
off_x = 0
off_y = 0

debug = 0
fps = 60
speed = 10
phys_steps = 100

# Environment
dt = 0.3

# Simulation
body_vel = 0.1

ball_angle = [90]
ball_size = 20
ball_vel = 0.07

square_angle = [-90]
square_size = 80
square_vel = 0.12

n_steps = 1000
log_name = ''

# Brain
eta_x_int = [0]
x_int_start = None

pi_eta_x_int = 0.0
pi_x_int = 0.5
p_x_int = 1.0
pi_prop = 1.0
pi_vis = 0.3

lambda_int = 1.0

lr_int = 1.0
lr_a = 1.0

n_orders = 2
n_objects = 3
n_policy = 1
n_tau = 15

gain_prior = 0.0
gain_evidence = 100.0
w_bmc = 5.0

# Body
start = [0]
lengths = [200]

joints = {}
joints['trunk'] = {'link': None, 'angle': start[0],
                   'size': (lengths[0], 50)}
n_joints = len(joints)

norm_polar = [-360.0, 360.0]
norm_cart = [-sum(lengths), sum(lengths)]
