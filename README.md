# Pysc2_replay_data_extraction
A tutorial of data extraction for starcraft II replays using Deepmind's starcraft II RL environment pysc2.

# Environment
Please follow [Pysc2 repo](https://github.com/deepmind/pysc2) to install pysc2 and other dependencies. I am currently running
and testing data_extraction.py at starcraft II version 4.1.1. Please note that sc2 replays are version dependent, execution will 
fail if replay was made with a different version of the game.

# Getting started
Clone or download this repo, and cd to the directory, run

```shell
python -m data_extraction --replay "path-to-replay" --norender --map_path "path-to-map" --step_mul 10 --fps 30
```
Additional flags:
```shell
--disable_fog   # disable fog of war, default False.
--fps   # frames per second to run the game, default 30.
--step_mul   # game steps per observation, default 10.
--observed_player   # which player to observe, default 1.
--helpfull   # see help on flags.
```

# Data
- The data will be download to ./replay_data/, screen and minimap features use sparse matrix to store only the nonzero elements and their position indices. Please check replay_data_visualize.ipynb for more details.
- Replay_example.7z is an example of replay data used in replay_data_visualize.ipynb.

# Troubleshooting
If you have any problem running script, feel free to leave a comment.
