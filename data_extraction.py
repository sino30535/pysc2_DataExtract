from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import platform
import sys
import time
import os
import csv

import mpyq
import six
from pysc2 import maps
from pysc2 import run_configs
from pysc2.env import sc2_env
from pysc2.lib import renderer_human
from pysc2.lib import stopwatch
from pysc2.lib import features
from pysc2.lib import point
from pysc2.lib import protocol
from pysc2.lib import remote_controller
from scipy import sparse

from absl import app
from absl import flags
from s2clientprotocol import sc2api_pb2 as sc_pb

FLAGS = flags.FLAGS

flags.DEFINE_float("fps", 30, "Frames per second to run the game.")
flags.DEFINE_integer("step_mul", 10, "Game steps per observation.")

flags.DEFINE_integer("screen_resolution", 84,
                     "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 64,
                     "Resolution for minimap feature layers.")

flags.DEFINE_bool("disable_fog", False, "Disable fog of war.")
flags.DEFINE_integer("observed_player", 1, "Which player to observe.")

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")

flags.DEFINE_string("map_path", None, "Override the map for this replay.")
flags.DEFINE_string("replay", None, "Name of a replay to show.")
flags.mark_flag_as_required("replay")

FLAGS(sys.argv)

def main(unused_argv):
    """Run SC2 to play a game or a replay."""
    stopwatch.sw.enabled = FLAGS.profile or FLAGS.trace
    stopwatch.sw.trace = FLAGS.trace

    if FLAGS.replay and not FLAGS.replay.lower().endswith("sc2replay"):
        sys.exit("Replay must end in .SC2Replay.")

    run_config = run_configs.get()

    interface = sc_pb.InterfaceOptions()
    interface.score = True
    interface.feature_layer.width = 24
    interface.feature_layer.resolution.x = FLAGS.screen_resolution
    interface.feature_layer.resolution.y = FLAGS.screen_resolution
    interface.feature_layer.minimap_resolution.x = FLAGS.minimap_resolution
    interface.feature_layer.minimap_resolution.y = FLAGS.minimap_resolution

    replay_data = run_config.replay_data(FLAGS.replay)
    start_replay = sc_pb.RequestStartReplay(
        replay_data=replay_data,
        options=interface,
        disable_fog=FLAGS.disable_fog,
        observed_player_id=FLAGS.observed_player)
    game_version = get_game_version(replay_data)
    print(game_version)

    with run_config.start(game_version=game_version) as controller:

        info = controller.replay_info(replay_data)
        print(" Replay info ".center(60, "-"))
        print(info)
        print("-" * 60)
        map_path = FLAGS.map_path or info.local_map_path
        print(map_path)
        if map_path:
            start_replay.map_data = run_config.map_data(map_path)
        controller.start_replay(start_replay)

        try:
            feat = features.Features(controller.game_info())
            while True:
                frame_start_time = time.time()
                controller.step(FLAGS.step_mul)
                obs = controller.observe()
                obs_t = feat.transform_obs(obs.observation)

                # Screen Features
                screen_features = ['height_map', 'visibility_map', 'creep', 'power', 'player_id', 'player_relative',
                                 'unit_type', 'selected', 'unit_hit_point', 'unit_hit_point_ratio', 'unit_energy',
                                 'unit_energy_ratio', 'unit_shield', 'unit_shield_ratio', 'unit_density',
                                 'unit_density_ratio', 'effects']
                # Minimap Features
                minimap_features = ['height_map', 'visibility_map', 'creep', 'camera', 'player_id', 'player_relative',
                                  'selected']
                # Other features
                other_features = ['player', 'game_loop', 'score_cumulative', 'available_actions', 'single_select',
                                'multi_select', 'cargo', 'cargo_slots_available', 'build_queue', 'control_groups']

                # Create replay data foldr
                data_folder = './replay_data/' + FLAGS.replay.split('\\')[-1][:10] + \
                            '_player_{}'.format(FLAGS.observed_player)
                if not os.path.exists(data_folder):
                    os.makedirs(data_folder)

                # Write screen features
                for i in range(len(screen_features)):
                    data_path = data_folder + "/" + 'screen_' + screen_features[i] + '.txt'
                    mode = 'a+' if os.path.exists(data_path) else 'w+'
                    with open(data_path, mode, newline='') as f:
                        writer = csv.writer(f)
                        writer.writerows(csr_matrix_to_list(sparse.csr_matrix(obs_t['screen'][i])))

                # Write minimap features
                for i in range(len(minimap_features)):
                    data_path = data_folder + "/" + 'minimap_' + minimap_features[i] + '.txt'
                    mode = 'a+' if os.path.exists(data_path) else 'w+'
                    with open(data_path, mode, newline='') as f:
                        writer = csv.writer(f)
                        writer.writerows(csr_matrix_to_list(sparse.csr_matrix(obs_t['minimap'][i])))

                # Write other features
                for i in other_features:
                    data_path = data_folder + "/" + i + '.txt'
                    mode = 'a+' if os.path.exists(data_path) else 'w+'
                    with open(data_path, mode, newline='') as f:
                        writer = csv.writer(f)
                        writer.writerows([obs_t[i]])

                # Write actions
                for action in obs.actions:
                    try:
                        func = feat.reverse_action(action).function
                        args = feat.reverse_action(action).arguments
                        try:
                            print(func, args)
                        except OSError:
                            pass

                        data_path = data_folder + "/" + 'action' + '.txt'
                        mode = 'a+' if os.path.exists(data_path) else 'w+'
                        with open(data_path, mode, newline='') as f:
                            writer = csv.writer(f)
                            writer.writerows([obs_t['game_loop'].tolist(), [func], [args]])
                    except ValueError:
                        pass
                if obs.player_result:
                    break
                time.sleep(max(0, frame_start_time + 1 / FLAGS.fps - time.time()))
        except KeyboardInterrupt:
            pass
        print("Score: ", obs.observation.score.score)
        print("Result: ", obs.player_result)

    if FLAGS.profile:
        print(stopwatch.sw)


def get_game_version(replay_data):
    replay_io = six.BytesIO()
    replay_io.write(replay_data)
    replay_io.seek(0)
    archive = mpyq.MPQArchive(replay_io).extract()
    metadata = json.loads(archive[b"replay.gamemetadata.json"].decode("utf-8"))
    version = metadata["GameVersion"]
    return ".".join(version.split(".")[:-1])


def csr_matrix_to_list(sparse_matrix):
    result = [
        sparse_matrix.data.tolist(),
        sparse_matrix.indices.tolist(),
        sparse_matrix.indptr.tolist()
    ]
    return result


def entry_point():  # Needed so setup.py scripts work.
    app.run(main)


if __name__ == "__main__":
    app.run(main)