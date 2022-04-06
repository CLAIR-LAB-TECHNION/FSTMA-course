import numpy as np


class PhysicsUtils:
    """
    A collection of tools based on a given `pettingzoo.mpe` environment's physical attributes. These tools do not
    account for `max_speed` limitations or collision mechanics. They also apply exactly one selected agent. Note that
    some tools work with scalars only, while others can work with numpy arrays.
    """

    def __init__(self, env, agent_selection=0):
        """
        create a utility instance for a given environment
        :param env: the environment for which the utility will apply
        :param agent_selection: the specific agent for whom to make the calculations (default 0).
        """
        self.env = env
        self.agent_selection = agent_selection

    ##################################
    # World Attributes as Properties ##########################
    # 1. provides easy access to world attributes             #
    # 2. will update if the environment configurations change #
    ###########################################################

    @property
    def max_force(self):
        """
        The maximal force that can be applied in any direction.
        :return: a non-negative float
        """
        return self.agent.u_range

    @property
    def a(self):
        """
        The selected agent acceleration
        :return: a float
        """
        return self.agent.accel or 5.0  # defaults to 5 if accel is None

    @property
    def dt(self):
        """
        The amount of time that passes at each environment step. Can be thought of as the time resolution.
        :return: a non-negative float
        """
        return self.world.dt

    @property
    def dd(self):
        """
        The drag. multiplies the speed at each time step
        :return: a value in [0, 1]
        """
        return 1 - self.world.damping

    @property
    def m(self):
        """
        the selected agent's mass.
        :return: a non-negative integer
        """
        return self.agent.mass

    @property
    def world(self):
        """
        The environment `World` object representing the current state.
        :return: the environment's `pettingzoo.mpe._mpe_utils.core.World` object.
        """
        return self.env.unwrapped.world

    @property
    def agent(self):
        """
        the selected agent in the world
        :return: the world's `pettingzoo.mpe._mpe_utils.core.Agent` at index `self.agent_selection`
        """
        return self.world.agents[self.agent_selection]

    #################
    # Physics Tools #
    # ###############

    def get_next_v(self, v, force):
        """
        get the velocity of the selected agent at the next time step.
        :param v: the current velocity
        :param force: the force to be applied at the next step
        :return: the new velocity after applying the given `force` to an agent moving at velocity `v`
        """
        return v * self.dd + ((force * self.a) / self.m) * self.dt

    def get_next_x(self, x, v, force):
        """
        get the position of the selected agent at the next time step.
        :param x: the current position
        :param v: the current velocity
        :param force: the force to be applied at the next step
        :return: the new position after applying the given `force` to an agent at `x` moving at velocity `v`
        """
        # must calculate next_v anyway. use function that returns both
        new_x, _ = self.get_next_xv(x, v, force)
        return new_x

    def get_next_xv(self, x, v, force):
        """
        get the position and velocity of the selected agent at the next time step.
        :param x: the current position
        :param v: the current velocity
        :param force: the force to be applied at the next step
        :return: a tuple (next_x, next_v) -- the new position and velocity after applying the given `force` to an agent
                 at `x` moving at velocity `v`.
        """
        # calculate next velocity
        new_v = self.get_next_v(v, force)

        # calculate next position using the new velocity
        new_x = x + new_v * self.dt

        return new_x, new_v

    def force_to_stop(self, v):
        """
        calculates the required force to stop an agent moving at a given velocity at the next time step.
        :param v: the agent's velocity
        :return: the force (direction and magnitude) to apply in order to stop
        """
        return ((-v) * self.dd * self.m) / (self.dt * self.a)

    def force_to_target(self, x, v, x_target):
        """
        calculates the required force to get an agent to a specified position at the next time step.
        :param x: the agent's current position
        :param v: the agent's current position
        :param x_target: the desired location to reach
        :return: the force (direction and magnitude) to apply in order to get to `x_target` from `x` with initial
                 velocity `v`.
        """
        return ((x_target - x - v * self.dd * self.dt) * self.m) / (self.a * (self.dt ** 2))

    def constant_force(self, x_0, v_0, f, t):
        """
        calculates the position and velocity of an agent after applying force `f` for `t` time steps.
        :param x_0: the initial location.
        :param v_0: the initial velocity.
        :param f: the constant force to apply at each time step.
        :param t: the time-step of the desired position and velocity.
        :return: a tuple (x_t, v_t), assuming a constant force `f` was applied at each time step before `t`.
        """
        # collect useful values required for the calculation
        ddt = self.dd ** t
        dd_denom = (self.dd - 1)
        ddi_sum_0_to_t_minus_1 = (ddt - 1) / dd_denom
        ddi_sum_1_to_t = self.dd * ddi_sum_0_to_t_minus_1
        fatm = (f * self.a * self.dt) / self.m
        fat2m_denom = (fatm * self.dt) / dd_denom

        # get v_t
        v_t = v_0 * ddt + fatm * ddi_sum_0_to_t_minus_1

        # calculate x_t
        x_t = x_0 + ddi_sum_1_to_t * (v_0 * self.dt + fat2m_denom) - t * fat2m_denom

        return x_t, v_t

    #######################
    # Axis Specific Tools ###############################################
    # The following functions are used to perform calculations per axis #
    # - simultaneously using numpy arrays                               #
    # - one axis value at a time (scalar)                               #
    # ###################################################################

    def can_stop(self, v):
        """
        check if an agent moving at a given velocity can stop at the next time step (i.e., next_v = 0).
        :param v: the agent's velocity
        :return: `True` iff an agent in the environment can stop using one action. if `v` is a numpy array, returns
                 an array of equal shape with `True` values for axes on which our agent can stop moving.
        """
        # check that the magnitude of the required force to stop the agent is less than the maximal force.
        return np.abs(self.force_to_stop(v)) <= self.max_force

    def max_force_in_dir(self, u):
        """
        returns maximal force in the direction of the given value.
        :param u: a target value
        :return: the maximal force in `u`'s direction. if `u` is a numpy array, returns an array of the same shape
                 with max_force values that have the same sign as their corresponding axis.
        """
        return np.sign(u) * self.max_force

    def two_moves_from_target(self, x, v, x_target):
        """
        checks if an agent can reach a given target and come to a complete stop within two time steps
        :param x: the agent's current position
        :param v: the agent's current velocity
        :param x_target: the desired location to reach
        :return: `True` iff the selected agent at position `x` with velocity `v` can stop at position `x_target` within
                 two time steps.
        """
        # check force required to reach the target
        f_to_target = self.force_to_target(x, v, x_target)

        # get the next
        next_v = self.get_next_v(v, f_to_target)

        # check valid force value and that the agent can stop given the next velocity.
        return (np.abs(f_to_target) <= self.max_force) & self.can_stop(next_v)

    #####################
    # Scalar Only Tools ##################
    # work only with scalar input values #
    # ####################################

    def can_stop_before_target(self, x, v, x_target):
        """
        checks if an agent at a given position with a given velocity has the ability to stop before passing a specified
        target position. This function only accept scalar values as input.
        :param x: the agent's current position in a single axis.
        :param v: the agent's current velocity in a single axis.
        :param x_target: the desired location to reach in a single axis.
        :return: `True` iff applying full force away from the target will cause the agent to stop before passing
                 `x_target`.
        """
        # can stop, then we are done
        if self.can_stop(v):
            return True

        # can't stop and on target
        if x == x_target:
            return False

        # going the other direction
        v_dir = np.sign(v)
        target_dir = np.sign(x_target)
        if v_dir != target_dir:
            return True

        # break as hard as possible and check if point is surpassed
        break_force = -v_dir * self.max_force
        next_x_full_break, next_v_full_break = self.get_next_xv(x, v, break_force)

        # target overtaken with full break.
        if abs(next_x_full_break) > abs(x_target):
            return False

        # recursive call to check if after the full break we meet one of the above exit conditions.
        return self.can_stop_before_target(next_x_full_break, next_v_full_break, x_target)

    def can_full_accelerate(self, v, x_target):
        """
        checks if accelerating with full force will cause the agent to surpass the target. Assumes the positions are
        in the agent's coordinate frame (i.e., the agent's position is (0, 0)). This function only accept scalar values
        as input.
        :param v: the agent's current velocity in a single axis.
        :param x_target: the desired location to reach in a single axis.
        :return: `True` iff the agent can accelerate with `max_force` towards `x_target` such that it still has the
                 opportunity to stop without passing `x_target` in the given axis.
        """
        full_acc = self.max_force_in_dir(x_target)
        next_x, next_v = self.get_next_xv(0, v, full_acc)

        # check that the target was not overshot and that the agent can stop before the target
        return (np.abs(next_x) <= abs(x_target)) & self.can_stop_before_target(next_x, next_v, x_target)
