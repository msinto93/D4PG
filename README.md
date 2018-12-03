# Distributed Distributional Deep Deterministic Policy Gradients (D4PG)
A Tensorflow implementation of a [**Distributed Distributional Deep Deterministic Policy Gradients (D4PG)**](https://arxiv.org/abs/1804.08617) network, for continuous control.

D4PG builds on the Deep Deterministic Policy Gradients (DDPG) approach ([paper](https://arxiv.org/pdf/1509.02971.pdf), [code](https://github.com/msinto93/DDPG)), making several improvements including the introduction of a distributional critic, using distributed agents running on multiple threads to collect experiences, prioritised experience replay (PER) and N-step returns.

![](http://wwdabney.gitlab.io/img/distributional_bellman.png)

Trained on [OpenAI Gym environments](https://gym.openai.com/envs).

This implementation has been successfully trained and tested on the [Pendulum-v0](https://gym.openai.com/envs/Pendulum-v0/), [BipedalWalker-v2](https://gym.openai.com/envs/BipedalWalker-v2/) and [LunarLanderContinuous-v2](https://gym.openai.com/envs/LunarLanderContinuous-v2/) environments. This code can however be run on any environment with a low-dimensional (non-image) state space and continuous action space.

**This currently holds the high score for the Pendulum-v0 environment on the [OpenAI leaderboard](https://github.com/openai/gym/wiki/Leaderboard#pendulum-v0)**

## Requirements
Note: Versions stated are the versions I used, however this will still likely work with other versions.

- Ubuntu 16.04 (Most (non-Atari) envs will also work on Windows)
- python 3.5
- [OpenAI Gym](https://github.com/openai/gym) 0.10.8 (See link for installation instructions + dependencies)
- [tensorflow-gpu](https://www.tensorflow.org/) 1.5.0
- [numpy](http://www.numpy.org/) 1.15.2
- [scipy](http://www.scipy.org/install.html) 1.1.0
- [opencv-python](http://opencv.org/) 3.4.0
- [imageio](http://imageio.github.io/) 2.4.1 (requires [pillow](https://python-pillow.org/))
- [inotify-tools](https://github.com/rvoicilas/inotify-tools/wiki) 3.14

## Usage
The default environment is 'Pendulum-v0'. To use a different environment simply change the `ENV` parameter in `params.py` before running the following files.

To train the D4PG network, run
```
  $ python train.py
```
This will train the network on the specified environment and periodically save checkpoints to the `/ckpts` folder.

To test the saved checkpoints during training, run
```
  $ python test_every_new_ckpt.py
```
This should be run alongside the training script, allowing to periodically test the latest checkpoints as the network trains. This script will invoke the `run_every_new_ckpt.sh` shell script which monitors the given checkpoint directory and runs the `test.py` script on the latest checkpoint every time a new checkpoint is saved. Test results are saved to a text file in the `/test_results` folder (optional).

Once we have a trained network, we can visualise its performance in the environment by running
```
  $ python play.py
```
This will play the environment on screen using the trained network and save a GIF (optional).

**Note:** To reproduce the best 100-episode performance of **-123.11 +/- 6.86** that achieved the top score on the ['Pendulum-v0' OpenAI leaderboard](https://github.com/openai/gym/wiki/Leaderboard#pendulum-v0), run
```
  $ python test.py
```
specifying the `test_params.ckpt_file` parameter in `params.py` as `Pendulum-v0.ckpt-660000`.

## Results
Result of training the D4PG on the 'Pendulum-v0' environment:

![](/video/Pendulum-v0.gif)

Result of training the D4PG on the 'BipedalWalker-v2' environment:

*To-Do*
![](/video/BipedalWalker-v2.gif)

Result of training the D4PG on the 'LunarLanderContinuous-v2' environment:

*To-Do*
![](/video/LunarLanderContinuous-v2.gif)

| **Environment**      | **Best 100-episode performance** | **Ckpt file** |
|----------------------|----------------------------------|---------------|
| Pendulum-v0          |  -123.11 +- 6.86                 | ckpt-660000   |

## To-do
- Train/test on further environments, including [Mujoco](http://www.mujoco.org/)

## References
- [A Distributional Perspective on Reinforcement Learning](http://wwdabney.gitlab.io/publication/distributional-perspective/)
- [Distributed Distributional Deterministic Policy Gradients](https://arxiv.org/abs/1804.08617)
- [OpenAI Baselines - Prioritised Experience Replay implementation](https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py)
- [OpenAI Baselines - Segment Tree implementation](https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py)
- [DeepMind TRFL Library - L2 Projection](https://github.com/deepmind/trfl/blob/master/trfl/dist_value_ops.py)
## License
MIT License
