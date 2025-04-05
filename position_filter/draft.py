from filterpy.common import Q_discrete_white_noise

q_noise = Q_discrete_white_noise(dim=2, dt=1, var=0.5**2, block_size=2)
print(q_noise)