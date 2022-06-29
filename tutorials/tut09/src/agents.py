import torch as T
import numpy as np
import torch.nn.functional as F
from src.networks import DqnMlp, Actor, Critic
from mac.agents import DecisionMaker
from src.replay_memory import ReplayBuffer
from src.utils import OUActionNoise


class DqnBase(DecisionMaker):
    def __init__(self, n_actions, input_dims, gamma=0.99, epsilon=1, lr=1e-4, 
                mem_size=10000, batch_size=16, eps_min=0.01, eps_dec=1e-4,
                replace=1000, algo=None, env_name=None, sensor_function=None, chkpt_dir='tmp/dqn'):
        super().__init__()
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_count = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0
        self.sensor_function = sensor_function or (lambda x: x)  # default to identity function

        self.memory = ReplayBuffer(self.mem_size, self.input_dims, self.n_actions)
    
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)

        return states, actions, rewards, states_, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_count == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()


class DqnDecisionMaker(DqnBase):
    def __init__(self, *args, **kwargs):
        super(DqnDecisionMaker, self).__init__(*args, **kwargs)

        self.q_eval = DqnMlp(self.input_dims, self.n_actions, self.lr,
                             name=self.env_name + '_' + self.algo + '_q_eval',
                             chkpt_dir=self.chkpt_dir)
        self.q_next = DqnMlp(self.input_dims, self.n_actions, self.lr,
                             name=self.env_name + '_' + self.algo + '_q_next',
                             chkpt_dir=self.chkpt_dir)

    def get_action(self, observation):
        with T.no_grad():
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()

        return action

    def get_training_action(self, observation):
        if np.random.random() > self.epsilon:
            with T.no_grad():
                state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
                actions = self.q_eval.forward(state)
                action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        if self.learn_step_counter < self.batch_size:
            self.learn_step_counter += 1
            return

        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()
        states, actions, rewards, states_, dones = self.sample_memory()

        # Deep Q learning update rule
        indices = np.arange(self.batch_size)
        q_pred = self.q_eval(states)[indices, actions]
        q_next = self.q_next(states_).max(dim=1)[0]
        q_next[dones] = 0.0
        q_target = rewards + self.gamma * q_next
        loss = self.q_eval.loss(q_pred, q_target).to(self.q_eval.device)

        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()


class DdpgDecisionMaker(DecisionMaker):
    def __init__(self, actor_lr, critic_lr, gamma, tau, n_actions, input_dims,
                 batch_size, mem_size, fc1_dim, fc2_dim, sensor_function=None,
                 env_name=None, chkpt_dir='tmp/dqn'):
        super().__init__()

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions  # the action is an n dimensional array of real numbers
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.mem_size = mem_size
        self.learn_step_counter = 0
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.sensor_function = sensor_function or (lambda x: x)  # default to identity function

        # init replay buffer
        self.memory = ReplayBuffer(input_dims, n_actions, mem_size)

        # init noise
        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        # init actor and actor_target networks
        self.actor = Actor(actor_lr, input_dims, fc1_dim, fc2_dim, n_actions, name='actor')
        self.actor_target = Actor(actor_lr, input_dims, fc1_dim, fc2_dim, n_actions, name='actor_target')

        # init critic and critic_target network
        self.critic = Critic(critic_lr, input_dims, fc1_dim, fc2_dim, n_actions, name='critic')
        self.critic_target = Critic(critic_lr, input_dims, fc1_dim, fc2_dim, n_actions, name='critic_target')

        self.update_network_parameters(tau=1)

    def get_action(self, state):
        # transfer the Actor network into an evaluation mode (relevant only when using Layer/Batch Norm)
        self.actor.eval()

        # converting the state into a PyTorch tensor and sending it to the actor device
        state = T.tensor([state], dtype=T.float).to(self.actor.device)

        # computing the deterministic actor mu
        mu = self.actor.forward(state).to(self.actor.device)

        # adding the noise for exploration
        noisy_mu = mu + T.tensor(self.noise(), dtype=T.float).to(self.actor.device)

        # transfer the actor network back to training mode
        self.actor.train()

        return noisy_mu.cpu().detach().numpy()[0]

    def get_training_action(self):
        pass

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        # collect parameters from networks
        actor_params = self.actor.named_parameters()
        actor_target_params = self.actor_target.named_parameters()
        critic_params = self.critic.named_parameters()
        critic_target_params = self.critic_target.named_parameters()

        # convert parameters into state dictionaries
        actor_state_dict = dict(actor_params)
        actor_target_state_dict = dict(actor_target_params)
        critic_state_dict = dict(critic_params)
        critic_target_state_dict = dict(critic_target_params)

        # iterate and update
        for item in actor_state_dict:
            actor_state_dict[item] = tau * actor_state_dict[item].clone() + \
                                     (1-tau) * actor_target_state_dict[item].clone()
        for item in critic_state_dict:
            critic_state_dict[item] = tau * critic_state_dict[item].clone() + \
                                      (1-tau) * critic_target_state_dict[item].clone()

        # load the new parameters as state dicts into the networks
        self.critic_target.load_state_dict(critic_state_dict)
        self.actor_target.load_state_dict(actor_state_dict)

    def store_transition(self, state, action, reward, state_, done):
        # appending the transition into the replay buffer memory
        self.memory.store_transition(state, action, reward, state_, done)

    def save_models(self):
        # saving the actor networks
        self.actor.save_checkpoint()
        self.actor_target.save_checkpoint()

        # saving the critic networks
        self.critic.save_checkpoint()
        self.critic_target.save_checkpoint()

    def load_models(self):
        # loading the actor networks
        self.actor.load_checkpoint()
        self.actor_target.load_checkpoint()

        # loading the critic networks
        self.critic.load_checkpoint()
        self.critic_target.load_checkpoint()

    def learn(self):
        # if memory not contains at least batch size transitions return without learning
        if self.learn_step_counter < self.batch_size:
            self.learn_step_counter += 1
            return

        # sample a transition batch from the memory buffer
        states, actions, rewards, states_, dones = self.memory.load_batch(self.batch_size)

        # transform to PyTorch tensors and send to device (same device for all so does not matter which)
        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        dones = T.tensor(dones).to(self.actor.device)

        # compute the next state actions
        target_actions = self.actor_target.forward(states_)
        critic_value_ = self.critic_target.forward(states_, target_actions)
        critic_value = self.critic.forward(states, actions)  # shape = (batch_size, 1)

        # set value for terminal states to zero and squeeze into a single dimension
        critic_value_[dones] = 0.0
        critic_value_ = critic_value_.view(-1)  # shape = (batch_size)

        # compute the critic target and reshape into batch size by 1
        target = rewards + self.gamma * critic_value_  # shape = (batch_size)
        target = target.view(self.batch_size, 1)  # shape = (batch_size, 1)

        # zeroing the critic gradients, compute the loss and backpropagation
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        # zeroing the actor gradients, compute the loss and backpropagation
        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(states, self.actor.forward(states))
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()
