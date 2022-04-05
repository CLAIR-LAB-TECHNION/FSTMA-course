from pettingzoo.mpe.simple_v2 import raw_env, make_env
from gym import spaces


class DiscreteNav():
    def __init__(self, pos_limit, pos_bins, vel_limit, vel_bins):
        self.pos_limit = pos_limit
        self.pos_bins = pos_bins
        self.vel_limit = vel_limit
        self.vel_bins = vel_bins
        self.pos_bin_size = 2 * pos_limit / pos_bins
        self.vel_bin_size = 2 * vel_limit / vel_bins
        self.env = make_env(raw_env)()

    def get_pos_idx(self, pos):
        if pos < - self.pos_limit:
            return -float('inf')
        elif pos > self.pos_limit:
            return float('inf')
        return (pos + self.pos_limit) // self.pos_bins

    def get_vel_idx(self, vel):
        if vel < - self.vel_limit:
            return -float('inf')
        elif vel > self.vel_limit:
            return float('inf')
        return (vel + self.vel_limit) // self.vel_bins

    def transform_state(self, state):
        self.state.agent.p_vel
