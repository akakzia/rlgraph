import numpy as np
from scipy.linalg import block_diag
from utils import get_idxs_per_relation, get_idxs_per_object
from collections import deque
from itertools import combinations, permutations

g_structure = list(combinations(np.arange(5), 2)) + list(permutations(np.arange(5), 2)) 

def get_ordered(goal, map):
  i_stacks = 10
  res = [] # will contain deques for different stacks
  for i in range(i_stacks, len(map)):
    base = g_structure[i][1]
    summit = g_structure[i][0]
    if goal[i] == 1.:
      if len(res) == 0:
        res.append(deque([base, summit]))
      else:
        j = 0
        added = False
        while not added and j < len(res):
          current_stack = res[j]
          base_current_stack = current_stack[0]
          summit_current_stack = current_stack[-1]
          if base == summit_current_stack:
            res[j].append(summit)
            added = True
            j = 0
          elif summit == base_current_stack:
            res[j].appendleft(base)
            added = True
            j = 0 
          j = j + 1
        if j == len(res):
          res.append(deque([base, summit]))
  finished = False
  flattened_res = res
  try:
    while len(flattened_res) > 1 and not finished:
        for i in range(len(flattened_res)-1):
                j = i
                while j < len(flattened_res):
                    j = j + 1
                    if flattened_res[j][0] == flattened_res[i][-1]:
                        flattened_res[i].pop()
                        flattened_res[i] = flattened_res[i] + flattened_res[j]
                        flattened_res.remove(flattened_res[j])
                        finished = False
                        if len(flattened_res) == 1:
                            return flattened_res
                    elif flattened_res[j][-1] == flattened_res[i][0]:
                        flattened_res[j].pop()
                        flattened_res[j] = flattened_res[j] + flattened_res[i]
                        flattened_res.remove(flattened_res[i])
                        finished = False
                        if len(flattened_res) == 1:
                            return flattened_res
                    else:
                        finished = True
    return flattened_res
  except IndexError:
      return []

class her_sampler:
    def __init__(self, args, reward_func=None):
        self.reward_type = args.reward_type
        self.replay_strategy = args.replay_strategy
        self.replay_k = args.replay_k
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + args.replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func
        self.continuous = args.algo == 'continuous'  # whether to use semantic configurations or continuous goals
        self.multi_criteria_her = args.multi_criteria_her
        self.obj_ind = np.array([np.arange(i * 3, (i + 1) * 3) for i in range(args.n_blocks)])

        if self.reward_type == 'per_object':
            self.semantic_ids = get_idxs_per_object(n=args.n_blocks)
        else:
            self.semantic_ids = get_idxs_per_relation(n=args.n_blocks)
        self.mask_ids = get_idxs_per_relation(n=args.n_blocks)

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions

        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}

        if not self.continuous:
            # her idx
            if self.multi_criteria_her:
                for sub_goal in self.semantic_ids:
                    her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
                    future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
                    future_offset = future_offset.astype(int)
                    future_t = (t_samples + 1 + future_offset)[her_indexes]
                    # Replace
                    future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
                    transition_goals = transitions['g'][her_indexes]
                    transition_goals[:, sub_goal] = future_ag[:, sub_goal]
                    transitions['g'][her_indexes] = transition_goals
            else:
                her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
                n_replay = her_indexes[0].size
                future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
                future_offset = future_offset.astype(int)
                future_t = (t_samples + 1 + future_offset)[her_indexes]

                # replace goal with achieved goal
                future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
                transitions['g'][her_indexes] = future_ag
                # to get the params to re-compute reward
            transitions['r'] = np.expand_dims(np.array([self.compute_reward_masks(ag_next, g) for ag_next, g in zip(transitions['ag_next'],
                                                        transitions['g'])]), 1)
        else:
            if self.multi_criteria_her:
                for sub_goal in self.obj_ind:
                    her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
                    future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
                    future_offset = future_offset.astype(int)
                    future_t = (t_samples + 1 + future_offset)[her_indexes]
                    # Replace
                    future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
                    transition_goals = transitions['g'][her_indexes]
                    transition_goals[:, sub_goal] = future_ag[:, sub_goal]
                    transitions['g'][her_indexes] = transition_goals
            else:
                # her idx
                her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
                future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
                future_offset = future_offset.astype(int)
                future_t = (t_samples + 1 + future_offset)[her_indexes]

                # replace goal with achieved goal
                future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
                transitions['g'][her_indexes] = future_ag
            transitions['r'] = np.expand_dims(np.array([self.reward_func(ag_next, g, None) for ag_next, g in zip(transitions['ag_next'],
                                                                                                transitions['g'])]), 1)

        return transitions

    def compute_reward_masks(self, ag, g):
        ordered_piles = get_ordered(g, g_structure)
        if len(ordered_piles) == 0:
            if self.reward_type == 'sparse':
                return (ag == g).all().astype(np.float32)
            elif self.reward_type == 'per_predicate':
                return (ag == g).astype(np.float32).sum()
            else:
                reward = 0.
                for subgoal in self.semantic_ids:
                    if (ag[subgoal] == g[subgoal]).all():
                        reward = reward + 1.
        else:
            # Test new reward ordered by piles
            reward = 0.
            treated_objects = []
            for pile in ordered_piles:
                for i in pile:
                    if (self.semantic_ids[i] == self.semantic_ids[i]).all() and i not in treated_objects:
                        reward = reward + 1.
                        treated_objects.append(i)
                    elif (self.semantic_ids[i] != self.semantic_ids[i]).all():
                        return reward
            for i, subgoal in enumerate(self.semantic_ids):
                if i not in treated_objects:
                    if (ag[subgoal] == g[subgoal]).all():
                        reward = reward + 1.

        return reward