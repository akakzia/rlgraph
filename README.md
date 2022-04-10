# Learning Self-Supervised Behaviors with Graph Attention-based Architectures

This repository contains the code associated to the *Learning Self-Supervised Behaviors with Graph Attention-based Architectures* paper published at the Workshop on Agent Learning in Open-Endedness (ALOE) at ICLR 2022.

**Abstract**
Although humans live in an open-ended world with endless challenges, they do not have to learn from scratch whenever they encounter a new task. Rather, they have access to a handful of previously learned skills, which they rapidly adapt to new situations. In artificial intelligence, autotelic agents—which are intrinsically motivated to represent and set their own goals—exhibit promising skill transfer capabilities. However, their learning capabilities are highly constrained by their policy and goal space representations. In this paper, we propose to investigate the impact of these representations. We study different implementations of autotelic agents using four types of Graph Neural Networks policy representations and two types of goal spaces, either geometric or predicate-based. We show that combining object-centered architectures that are expressive enough with semantic relational goals enables an efficient transfer between skills and promotes behavioral diversity. We also release our graph-based implementations to encourage further research in this direction.

<p align="center">
  <img src="https://i.ibb.co/zh7vmdk/graphs-v4.jpg" />
</p>

We model both the critic and the policy using four different GNN-based architectures: full graph networks, interaction networks, relation networks and deep sets. The code for each architecture can be found in the rl_modules/ folder. 

**Requirements**

* gym
* mujoco
* pytorch
* pandas
* matplotlib
* numpy

To reproduce the results, you need a machine with **24** cpus.


**Training Graph-based Autotelic Agents**

The following lines launch the training of our autotelic agents. When using semantic goal spaces, agents explore their behavioral space, discover new goals and attempt to learn them. When using continuous goal space, agents first select a class of goals based on the number of desired stacks. This selection is based on an LP-based Curriculum Learning which is shown to stabilize the learning procedure

```mpirun -np 24 python train.py --algo semantic```

```mpirun -np 24 python train.py --algo continuous```

To specify the GNN-based architecture to be used, refer to one of the following lines: 

```mpirun -np 24 python train.py --architecture full_gn```

```mpirun -np 24 python train.py --architecture interaction_network```

```mpirun -np 24 python train.py --architecture relation_network```

```mpirun -np 24 python train.py --architecture deep_sets```


Note: The folder configs/ contains the hyper-parameters used for both types of goal spaces. 

