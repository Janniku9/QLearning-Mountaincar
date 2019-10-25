# Solving Mountain Car Problem using Qlearning

This is my personal solution for the [openai gym mountain car problem](https://gym.openai.com/envs/MountainCar-v0/) i made in early 2019. I have two different approaches, the first one is using a Q-Table and the second one is using a neural network. The problem can be described as following:

> A car is on a one-dimensional track, positioned between two "mountains". The goal is to drive up the mountain on the right; however, the car's engine is not strong enough to scale the mountain in a single pass. Therefore, the only way to succeed is to drive back and forth to build up momentum.

<center>
<img src="https://raw.githubusercontent.com/Janniku9/QLearning-Mountaincar/master/img/mountaincar.gif" width="400"/>
</center>

The state S we observe consists of the position and the velocity (where the sign represents the direction the car is moving). We can choose 3 actions, apply force to the right, apply force to the left or do nothing. The reward we use in this example is simply the time it takes to reach the top (timeout after 200 frames).


## Q-table

[This solution](https://github.com/Janniku9/QLearning-Mountaincar/blob/master/qtable.py) uses a Table for choosing the best next action. First, we restrict the state space by discretizing the space of the position and the velocity. 

```
s = 200
position_range = np.linspace(low[0], high[0], s)
velocity_range = np.linspace(low[1], high[1], int(s/5))
```

We can then make a QTable with Dimensions position_values * velocity_values * number_of_actions and update it using the following policy:

<center>
<img src="https://raw.githubusercontent.com/Janniku9/QLearning-Mountaincar/master/img/qalgorithm.png" width="600"/>
</center>

Using somewhat good values for learning rate, discount factor and exploration rate, we can get following result:

<center>
<img src="https://raw.githubusercontent.com/Janniku9/QLearning-Mountaincar/master/img/qtable_reward.png" width="400"/>
</center>

The plot above shows the reward over the iterations of the algorithm. We can see that the first success happens around the 4000th epoch. The spikes in the graph most likely originated when the algorithm had found a way in which he had to go once less to the right side.

## DeepQ learning

In [this solution](https://github.com/Janniku9/QLearning-Mountaincar/blob/master/deepq.py) we use deepq learning to solve the problem. This time we use the network in [DeepQNetwork.py](https://github.com/Janniku9/QLearning-Mountaincar/blob/master/DeepQNetwork.py) instead of a table. 

@TODO
