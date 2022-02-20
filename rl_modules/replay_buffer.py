import threading
import numpy as np

"""
the replay buffer here is basically from the openai baselines code

"""


class ReplayBuffer:
    def __init__(self, env_params, buffer_size, sample_func, goal_sampler):
        self.env_params = env_params
        self.T = env_params['max_timesteps']
        self.size = buffer_size // self.T
        self.goal_sampler = goal_sampler

        # memory management
        self.sample_func = sample_func

        # self.current_size = 0

        self.current_sizes = np.array([0 for _ in range(5)])

        # create the buffer to store info
        self.buffer = {'obs': np.empty([5, self.size, self.T + 1, self.env_params['obs']]),
                       'ag': np.empty([5, self.size, self.T + 1, self.env_params['goal']]),
                       'g': np.empty([5, self.size, self.T, self.env_params['goal']]),
                       'actions': np.empty([5, self.size, self.T, self.env_params['action']]),
                       }

        self.goal_ids = np.zeros([self.size])  # contains id of achieved goal (discovery rank)
        self.goal_ids.fill(np.nan)

        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, episode_batch):
        batch_size = len(episode_batch)
        with self.lock:
            # idxs = self._get_storage_idx(inc=batch_size)

            for i, e in enumerate(episode_batch):
                idx = self._get_storage_idx(e['goal_class'])
                # store the informations
                self.buffer['obs'][e['goal_class'], idx[0], :, :] = e['obs']
                self.buffer['ag'][e['goal_class'], idx[0], :, :] = e['ag']
                self.buffer['g'][e['goal_class'], idx[0], :, :] = e['g']
                self.buffer['actions'][e['goal_class'], idx[0], :, :] = e['act']
                # self.goal_ids[e['goal_class'], idx[0], :, :] = e['goal_class']

    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffers = {}
        with self.lock:
            # for key in self.buffer.keys():
            #     temp_buffers[key] = self.buffer[key][:self.current_size]

            # Compute goal id proportions with respect to LP probas
            goal_id = int(self.goal_sampler.build_batch(batch_size)[0])
            # check if buffer of goal id contains episodes
            if self.current_sizes[goal_id] == 0:
                goal_id = np.where(self.current_sizes > 0)[0]
                goal_id = np.random.choice(goal_id)
            for key in self.buffer.keys():
                temp_buffers[key] = self.buffer[key][goal_id, :self.current_sizes[goal_id], :, :]
                

            # buffer_ids = []
            # for g in goal_ids:
            #     buffer_ids_g = np.argwhere(self.goal_ids == g).flatten()
            #     if buffer_ids_g.size == 0:
            #         buffer_ids.append(np.random.choice(range(self.current_size)))
            #     else:
            #         buffer_ids.append(np.random.choice(buffer_ids_g))
            # buffer_ids = np.array(buffer_ids)
            # for key in self.buffer.keys():
            #     temp_buffers[key] = self.buffer[key][buffer_ids]

        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]


        # sample transitions
        transitions = self.sample_func(temp_buffers, batch_size)
        return transitions

    def _get_storage_idx(self, g_id, inc=None):
        inc = inc or 1
        if self.current_sizes[g_id] + inc <= self.size:
            idx = np.arange(self.current_sizes[g_id], self.current_sizes[g_id] + inc)
        elif self.current_sizes[g_id] < self.size:
            overflow = inc - (self.size - self.current_sizes[g_id])
            idx_a = np.arange(self.current_sizes[g_id], self.size)
            idx_b = np.random.randint(0, self.current_sizes[g_id], overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_sizes[g_id] = min(self.size, self.current_sizes[g_id] + inc)
        if inc == 1:
            idx = [idx[0]]
        return idx
