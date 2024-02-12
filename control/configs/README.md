# Configuration Files


## Environment Configuration

### Environment
- **Time limit** – The time limit is a float value corresponding to the number of seconds the robot is allowed to take in navigating to its goal before the simulation is ended. This value should be adjusted based on the simulation configurations.
- **Time step** – The time step is a float value corresponding to the number of seconds between each simulation update. At each time step, the robot and each of the humans chooses a new action according to its policy. It is recommended that this value be set to 0.25 seconds.
- **Validation size** – The validation size is an integer value corresponding to the number of episodes that our run in the validation stage while training the RL policy.
- **Test size** – The test size is an integer value corresponding to the number of episode that are used to evaluate the trained policy.

### Reward
- **Success reward** – The success reward is the positive reward the robot receives once it reaches its goal. (Default: 1.0)
- **Collision penalty** – The human collision penalty is the negative reward the robot receives for colliding with a human. (Default: -0.25)
- **Discomfort distance** – The discomfort distance is the minimum distance between the robot and the human that is considered comfortable.  (Default: 0.1)
- **Discomfort penalty factor** – When the robot is within the discomfort distance, it receives a negative reward based on how close the robot is to the human.  (Default: 0.5)
- **Time penalty** – The time penalty is the negative reward the robot receives at each time step. (Default: 0.0)
- **Progress reward** – The progress reward is the negative reward the robot receives for making progress towards its goal at each step. (Default: 0.0)
- **Goal radius** – The robot is considered to have reached the goal once it is within the goal radius. (Default: 0.3)

### Simulation
- **Room width** – Width of the simulated room in meters.
- **Room height** – Height of the simulated room in meters.
- **Default number of humans** – If the number of humans is not being randomized, the default number of humans are placed according to the human generating scheme.
- **Maximum number of humans** – If the number of humans is being randomized, the number of humans is sampled from a uniform distribution with the minimum and maximum values specified.
- **Minimum number of humans** –  If the number of humans is being randomized, the number of humans is sampled from a uniform distribution with the minimum and maximum values specified.
- **Maximum density of humans** – If the number of humans is being randomized based on the amount of open space, the number of humans is sampled from a uniform distribution whose minimum and maximum values are determined by the amount of open space and the minimum and maximum densities specified.
- **Minimum density of humans** – If the number of humans is being randomized based on the amount of open space, the number of humans is sampled from a uniform distribution whose minimum and maximum values are determined by the amount of open space and the minimum and maximum densities specified.
- **Randomize number of humans** – If this boolean value is set to true, the number of humans will be sampled from a uniform distribution based on the minimum and maximum number of humans specified.
- **Randomize density of humans** – If randomize number of humans is set to false and this boolean value is set to true, the number of humans will be sampled from a uniform distribution based on the minimum and maximum density of humans specified. If both boolean values are set to false, the default number of humans will be used.
- **End on collision** – The simulation will end when the robot collides with a human, object, or wall if this boolean value is set to true.
- **Plan human path** – A global path will be generated for all of the humans to follow from their initial position to their goal position if this boolean value is set to true. Note that this can significantly increase the time required for training/evaluation.
- **Perpetual** – The humans will be given a new goal position after reaching their goal to allow for perpetual motion if this boolean value is set to true. Note that perpetual motion is not intended to be used in combination with human path planning.
- **Pedestrian crossing scenario** – Human crossing scneario used to capture come primitive interaction between the robot and surrounding pedestrians (Values: circle, following, passing, crossing, or random.)
- **Randomness** – If this boolean value is set to true, the pedestrian crossing scenario will be randomly chosen.

### Waypoints
This section is only used for the hardware experiments.

### Humans
- **Visible** – Each human is visible to the robot and all other humans if this boolean value is set to true.
- **Policy** – The actions of the humans are determined by its policy. (Default: orca)
- **Sensor** – The humans perceive the environment based on their sensor. The only sensor that is currently implemented is called "coordinates" and allows the humans to view the position, velocity, and radius of visible agents.
- **Observability** – The humans can either have full observability of all visible agents in the environment, or they can only view unobstructed agents within their detection range. (Values: full, partial)
- **Detection range** – The detection range is how far the humans can view another agent.
- **Default radius** – If the radius of the humans is not being randomized, all of the humans have the default radius.
- **Minimum radius** – If the radius of the humans is being randomized, the radius of each human is sampled from a uniform distribution with the minimum and maximum values specified.
- **Maximum radius** – If the radius of the humans is being randomized, the radius of each human is sampled from a uniform distribution with the minimum and maximum values specified.
- **Default preferred velocity** –  If the preferred velocity of the humans is not being randomized, all of the humans have the default preferred velocity.
- **Minimum preferred velocity** – If the preferred velocity of the humans is being randomized, the preferred velocity of each human is sampled from a uniform distribution with the minimum and maximum values specified.
- **Maximum preferred velocity** – If the preferred velocity of the humans is being randomized, the preferred velocity of each human is sampled from a uniform distribution with the minimum and maximum values specified.
- **Default neighbor distance** – If the neighbor distance (ORCA parameter) of the humans is not being randomized, all of the humans have the default neighbor distance.
- **Minimum neighbor distance** – If the neighbor distance of the humans is being randomized, the neighbor distance of each human is sampled from a uniform distribution with the minimum and maximum values specified.
- **Maximum neighbor distance** – If the neighbor distance of the humans is being randomized, the neighbor distance of each human is sampled from a uniform distribution with the minimum and maximum values specified.
- **Default time horizon** – If the time horizon (ORCA parameter) of the humans is not being randomized, all of the humans have the default time horizon.
- **Minimum time horizon** – If the time horizon of the humans is being randomized, the time horizon of each human is sampled from a uniform distribution with the minimum and maximum values specified.
- **Maximum time horizon** – If the time horizon of the humans is being randomized, the time horizon of each human is sampled from a uniform distribution with the minimum and maximum values specified.
- **Randomize radius** – If this boolean value is set to true, the radius of each human will be sampled from a uniform distribution. If it is set to false, the default radius will be assigned to each human.
- **Randomize preferred velocity** – If this boolean value is set to true, the preferred velocity of each human will be sampled from a uniform distribution. If it is set to false, the default preferred velocity will be assigned to each human.
- **Randomize neighbor distance** – If this boolean value is set to true, the neighbor distance of each human will be sampled from a uniform distribution. If it is set to false, the default neighbor distance will be assigned to each human.
- **Randomize time horizon** – If this boolean value is set to true, the time horizon of each human will be sampled from a uniform distribution. If it is set to false, the default time horizon will be assigned to each human.

### Robot
- **Visible** – The robot is visible to the humans if this boolean value is set to true.
- **Policy** – If the robot is using the reinforcement learning policy, this parameter can be set to none. The robot can also be controlled using the ORCA policy.
- **Sensor** – The robot perceives the environment based on its sensor. The only sensor that is currently implemented is called "coordinates" and allows the robot to view the position, velocity, and radius of visible agents.
- **Observability** – The robot can either have full observability of all visible agents in the environment, or it can only view unobstructed agents within its detection range. (Values: full, partial)
- **Detection range** – The detection range is how far the robot can view another agent.
- **Radius** – The radius of the robot (assuming it can be represented as a circle).
- **Preferred velocity** – The preferred velocity of the robot (its maximum speed and speed it travels when unobstructed).



## Policy Configuration
### Reinforcement Learning (RL)
- **Gamma** – Gamma is the discount factor used in computing the value function. (Default: 0.9)

### Action Space
- **Kinematics** – The kinematics model used to control simulated model determines the format of the action provided by policy. The model must either be "holonomic" or "non-holonomic." A holonomic robot receives actions as x and y velocites, while a non-holonomic robot receives actions as linear and angular velocities.
- **Speed samples** – The action space is discretized into n speeds exponentially spaced in the range (0, v_pref] and m rotations evenly spaced in the range [0,2pi), where n is the number of speed samples and m is the number of rotation samples.
- **Rotation samples** – The action space is discretized into n speeds exponentially spaced in the range (0, v_pref] and m rotations evenly spaced in the range [0,2pi), where n is the number of speed samples and m is the number of rotation samples.
- **Query environment** – If set to true, the robot predicts the next state of the humans/objects and the rewards by querying the simulation environment. Otherwise, the next state and reward are estimated without using the simulation environment.

### Reward Function
- **Adjust discomfort distance** – If this boolean value is set to true, an uncertainty-dependent discomfort distance is used in the reward function of the RL policy.
- **Discomfort distance slope** – If eps is the uncertainty associated with a pedestrian, the discomfort distance assigned to this person is d = a * eps + b, where a is the discomfort distance slope and b is the intercept.
- **Discomfort distance intercept** – If eps is the uncertainty associated with a pedestrian, the discomfort distance assigned to this person is d = a * eps + b, where a is the discomfort distance slope and b is the intercept.

### Value Network
- **MLP1 dimensions** – Hidden layer sizes of the first set of multilayer perecptrons (MLPs) used by our policy.
- **MLP2 dimensions** – Hidden layer sizes of the second set of multilayer perecptrons (MLPs) used by our policy.
- **MLP3 dimensions** – Hidden layer sizes of the third set of multilayer perecptrons (MLPs) used by our policy.
- **Attention dimensions** – Hidden layer sizes of the set of multilayer perecptrons (MLPs) that compute the attention scores used by our policy.
- **With global state** – If the global state is used, the mean of the first set of MLPs will be used in computing the attention score.

### Safety Policy
This section is only used for the hardware experiments.



## Training Configuration
### Trainer
- **Batch size** – Number of batches of state transitions used for training the reinforcement learning (RL) policy.
- **Maximum agents** – If max agents is set to -1, there is no maximum allowable number of agents.

#### Imitation Learning
- **IL episodes** – Number of imitation learning (IL) episodes ran per epoch before training the RL policy in order to initialize the data buffer.  (Default: 3000)
- **IL policy** – Policy used to populate data buffer with state transitions. (Default: orca)
- **IL epochs** – Number of imitation learning (IL) epochs used to collect data to initialize the data buffer. (Default: 50)
- **IL learning rate** – Learning rate used when running the imitation learning (IL) policy. (Default: 0.01)
- **Safety space** – If the ORCA policy is used for imitation learning and the robot is not visible to the humans, adding safety space improves ORCA performance. (Default: 0.15)
- **Human randomness** – This float indicates the maximum epsilon value of the randomized ORCA policy used to train the imitation learning policy (Default: 0.0).  

### Reinforcement Learning
- **RL learning rate** – Learning rate used when training the reinforcement learning (IL) policy. (Default: 0.001)
- **Training batches** – Number of batches to train at the end of each training episode. (Default: 100)
- **Training episodes** – Number of training episodes used in the outer loop. (Default: 12000)
- **Sample episodes** – Number of training episodes sampled. (Default: 1)
- **Target update interval** – The target network used to update the value network is updated every n episodes, where n is the target update interval. (Default: 50)
- **Evaluation interval** – Validation is performed every n episodes, where n is the evaluation interval. (Default: 1000)
- **Capacity** – The capacity is the amount of data that the data buffer can hold. (Default: 50 * 2000 episodes = 10000)
- **Epsilon start** – The initial epsilon value used in the epsilon-greedy policy. (Default: 0.5)
- **Epsilon end** – The final epsilon value used in the epsilon-greedy policy. (Default: 0.1)
- **Epsilon decay** – The epsilon value used in the epsilon-greedy policy decays from the initial value to the final over n episodes, where n is the epsilon decay. (Default: 0.1)
- **Checkpoint interval** – The model is saved every n episodes, where n is the checkpoint interval. Training can be resumed from this checkpoint. (Default: 1000)
- **Human randomness start** – This float indicates the initial maximum epsilon value of the randomized ORCA policy used to train the RL policy (Default: 0.0).  
- **Human randomness end** – This float indicates the final maximum epsilon value of the randomized ORCA policy used to train the RL policy (Default: 0.5).  
- **Human randomness step** – This float indicates how much the maximum epsilon value is incremented each time the value is increased durnig training (Default: 0.1).  
