import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(x, scores, epsilons, filename):
	fig = plt.figure()
	ax = fig.add_subplot(111, label='1')
	ax2 = fig.add_subplot(111, label='2', frame_on=False)

	ax.plot(x, epsilons, color='C0')
	ax.set_xlabel('Training Steps', color='C0')
	ax.set_ylabel('Epsilon', color='C0')
	ax.tick_params(axis='x', color='C0')
	ax.tick_params(axis='y', color='C0')

	N = len(scores)
	running_avg = np.empty(N)
	for t in range(N):
		running_avg[t] = np.mean(scores[np.max([0, t - 100]):(t + 1)])

	ax2.scatter(x, running_avg, color='C1')
	ax2.axes.get_xaxis().set_visible(False)
	ax2.yaxis.tick_right()
	ax2.set_ylabel('Score', color='C1')
	ax2.yaxis.set_label_position('right')
	ax2.tick_params(axis='y', color='C1')

	plt.savefig(filename)


def plot_multi_taxi_learning_curve(total_rewards, agents_epsilons=None, filename=None):
	# extract agents names
	first_episode_rewards = total_rewards[0]
	first_iteration = first_episode_rewards[0]
	agents_names = list(first_iteration.keys())
	agents_rewards = {}
	for name in agents_names:
		agents_rewards[name] = []

	# rearrange rewards as per agent
	for episode in range(len(total_rewards)):
		episode_rewards = total_rewards[episode]
		for i in range(len(episode_rewards)):
			for agent_id in episode_rewards[i]:
				agents_rewards[agent_id].append(episode_rewards[i][agent_id])

	# compute rewards running average
	agents_avg_rewards = {}
	n = len(agents_rewards[agents_names[0]])
	for agent_id in agents_names:
		agents_avg_rewards[agent_id] = []
		for t in range(n):
			agents_avg_rewards[agent_id].append(np.mean(agents_rewards[agent_id][np.max([0, t - 100]):(t + 1)]))

	if agents_epsilons is None:
		agents_epsilons = {}
		for agent_id in agents_names:
			agents_epsilons[agent_id] = [0] * n

	fig = plt.figure(figsize=(13, 7))
	ax = fig.add_subplot(211, label='1')
	ax2 = fig.add_subplot(212, label='2')

	for agent_id in agents_names:
		ax.plot(range(n), agents_epsilons[agent_id], label=agent_id)
	ax.set_xlabel('Training Steps')
	ax.set_ylabel('Epsilon')
	ax.tick_params(axis='y')
	ax.axes.get_xaxis().set_visible(False)
	ax.legend()

	for agent_id in agents_names:
		ax2.plot(range(n), agents_avg_rewards[agent_id], label=agent_id)

	ax2.set_ylabel('Average reward')
	ax2.tick_params(axis='y')
	ax2.legend()

	if filename is not None:
		plt.savefig(filename + '.png', dpi=200)

	plt.show()


class OUActionNoise:
	def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
		self.mu = mu
		self.sigma = sigma
		self.theta = theta
		self.dt = dt
		self.x0 = x0
		self.reset()

	def __call__(self):
		x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
			self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
		self.x_prev = x
		return x

	def reset(self):
		self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)









