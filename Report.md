[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# Report for Project 2: Continuous Control for 20 Agents (Version 2)
    
## Introduction

In this `Report.md`, you can see an implementation for the second project of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

![Trained Agent][image1]
## Implementation specs

### 1. Summary

I implement the [Twin Delayed Deep Deterministic policy gradient algorithm (TD3)](https://arxiv.org/abs/1802.09477)

--------

### 2. Details

#### 2-1 Concepts


I refer to the [reference site](https://spinningup.openai.com/en/latest/algorithms/td3.html) and the [paper](https://arxiv.org/abs/1802.09477) for the concepts

While DDPG can achieve great performance sometimes, it is frequently brittle with respect to hyperparameters and other kinds of tuning. A common failure mode for DDPG is that the learned Q-function begins to dramatically overestimate Q-values, which then leads to the policy breaking, because it exploits the errors in the Q-function. Twin Delayed DDPG (TD3) is an algorithm that addresses this issue by introducing below critical tricks.

##### 2-1-1 Clipped Double-Q Learning

TD3 learns two Q-functions instead of one (hence “twin”), and uses the smaller of the two Q-values to form the targets in the Bellman error loss functions

- **Target policy smoothing** : Actions used to form the Q-learning target are based on the target policy, $\mu_{\theta_{\text{targ}}}$, but **with clipped noise added on each dimension of the action**. **After adding the clipped noise, the target action is then clipped to lie in the valid action range** (all valid actions, a, satisfy $a_{Low} \leq a \leq a_{High}$). The target actions are thus:

    $$a'(s') = \text{clip}\left(\mu_{\theta_{\text{targ}}}(s') + \text{clip}(\epsilon,-c,c), a_{Low}, a_{High}\right), \;\;\;\;\; \epsilon \sim \mathcal{N}(0, \sigma)$$

     **Target policy smoothing essentially serves as a regularizer for the algorithm**. It addresses a particular failure mode that can happen in DDPG: **if the Q-function approximator develops an incorrect sharp peak for some actions, the policy will quickly exploit that peak and then have brittle or incorrect behavior.** This can be averted by **smoothing out the Q-function over similar actions**, which target policy smoothing is designed to do.
- **clipped double-Q learning** : Both Q-functions use a single target value $y(r, s', d)$, **calculated using whichever of the two Q-functions gives a smaller target value**

    $$y(r,s',d) = r + \gamma (1 - d) \min_{i=1,2} Q_{\phi_{i, \text{targ}}}(s', a'(s')),$$

    and then both are learned by regressing to this target:

    ![https://spinningup.openai.com/en/latest/_images/math/7d5c18f49a242cc3eec554f717fe4f3bfc119bab.svg](https://spinningup.openai.com/en/latest/_images/math/7d5c18f49a242cc3eec554f717fe4f3bfc119bab.svg)

    ![https://spinningup.openai.com/en/latest/_images/math/cd73726a8a3845ade467aed57714912f868f6b36.svg](https://spinningup.openai.com/en/latest/_images/math/cd73726a8a3845ade467aed57714912f868f6b36.svg)

    Using the smaller Q-value for the target, and regressing towards that, helps fend off overestimation in the Q-function.

##### 2-1-2 Delayed Policy Updates : 

**TD3 updates the policy (and target networks) less frequently than the Q-function**. The paper recommends **one policy update for every two Q-function updates**.

Policy is learned just by maximizing $Q_{\phi_1}$:

$$\max_{\theta} \underset{s \sim {\mathcal D}}{{\mathrm E}}\left[ Q_{\phi_1}(s, \mu_{\theta}(s)) \right]$$

which is pretty much unchanged from DDPG. However, in TD3, the **policy is updated less frequently than the Q-functions are**. This helps damp the volatility that normally arises in DDPG because of how a policy update changes the target.

##### 2-1-3 Pseudocode

![https://spinningup.openai.com/en/latest/_images/math/b7dfe8fa3a703b9657dcecb624c4457926e0ce8a.svg](https://spinningup.openai.com/en/latest/_images/math/b7dfe8fa3a703b9657dcecb624c4457926e0ce8a.svg)

------------
#### 2-2. Networks

The network structure is as follows:

##### 2-2-1. Actor

state -> BatchNorm -> Linear(state_size, 256) -> BatchNorm -> LeakyRelu -> Linear(256, 128) -> BatchNorm -> LeakyRelu -> Linear(128, action_size) -> tanh

##### 2-2-2. Critic

state -> BatchNorm -> Linear(state_size, 256) -> Relu -> (concat with action) -> Linear(256+action_size, 128) -> Relu -> Linear(128, 1) 

#### 2-3. Hyperparameters

Agent hyperparameters are passed as constructor arguments to `Agent`.  The default values, used in this project, are:

| parameter    | value  | description                                                                   |
|--------------|--------|-------------------------------------------------------------------------------|
| BUFFER_SIZE  | 1e6    | Number of experiences to keep on the replay memory for the TD3                |
| BATCH_SIZE   | 128    | Minibatch size used at each learning step                                     |
| GAMMA        | 0.99   | Discount applied to future rewards                                            |
| TAU          | 1e-3   | Scaling parameter applied to soft update                                      |
| LR_ACTOR     | 1e-3   | Learning rate for actor used for the Adam optimizer                           |
| LR_CRITIC    | 1e-3   | Learning rate for critic used for the Adam optimizer                          |
| NUM_LEARN    | 10     | Number of learning at each step                                               |
| NUM_TIME_STEP| 20     | Every NUM_TIME_STEP do update                                                 |
| EPSILON      | 4      | Epsilon to noise of action                                                    |
| EPSILON_DECAY| 2e-6   | Epsilon decay to noise epsilon of action                                      |
| POLICY_DELAY | 3      | Delay for policy update (TD3)                                                 |

Training hyperparameters are passed to the training function `train` of `Agent`, defined below.  The default values are:

| parameter                     | value            | description                                                             |
|-------------------------------|------------------|-------------------------------------------------------------------------|
| n_episodes                    | 3000             | Maximum number of training episodes                                     |
| max_t                         | 3000             | Maximum number of steps per episode                                     |


-----------

### 3. Result and Future works

#### 3-1. Reward

![Reward](https://user-images.githubusercontent.com/73100569/126519350-071a4af5-5d5b-43e3-b98a-e065bd7c68f5.png)


Here x-axis is the episode and y-axis is the reward. Environment solved in 167 episodes. You can see relatively stable learning since TD3 is the improved version of DDPG.

#### 3-2. Future works

1. Parameters tuning for TD3. 
2. Implement this project by other algorithms like **PPO(Proximal Policy Optimization)** which is on-policy algorithm or **SAC(Soft Actor Critic)** which is off policy with entropy maximization to enable stability and exploration.
   