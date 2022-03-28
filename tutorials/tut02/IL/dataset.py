from torch.utils.data import Dataset


class ImitationLearningDataset(Dataset):
    def __init__(self, trajectories, prep_fn=None):
        """
        Creates a dataset
        :param obs_action_paris: an array like object of shape (N, 2) where the first column is the observations and the
                                 second is the corresponding action
        :param prep_fn: a f:obs --> torch.Tensor that preprocesses a single observation.
        """
        super().__init__()
        # merge trajectories into a single dataset of observation-action paris
        self.obs_action_paris = []
        for trajectory in trajectories:
            self.obs_action_paris.extend(list(zip(trajectory.observations, trajectory.actions)))

        # if no preprocessing function is given, use the identity function
        if prep_fn is None:
            self.prep_fn = lambda x: x
        else:
            self.prep_fn = prep_fn

    def __getitem__(self, index: int):
        """
        Returns a preprocessed sample observation and action label.
        :param index: Sample index.
        :return: A tuple (sample, label) where `sample` is preprocessed observation and its action label.
        Raises a ValueError if index is out of range.
        """
        state, action = self.obs_action_paris[index]

        return self.prep_fn(state), action

    def __len__(self):
        """
        :return: Number of samples in this dataset.
        """
        return len(self.obs_action_paris)

