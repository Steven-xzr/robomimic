"""
由raw hdf5数据集生成部分信息（不包括点云和segmentation），在世界坐标系下：
- 保存每个episode的每个timestep的rgb(image+video), depth(npy+video)
- 保存每个episode的每个timestep的end effector的位置和姿态，以及gripper的位置
- 保存每个episode的metadata(episode name, num_samples, trajectory_length, robot_base_pos, camera_names, camera_info, camera_height, camera_width, range_of_workspace_robot_base, end_of_subtasks, subtask_contact_flags)
- camra_info 相机外参在世界坐标系下
- 保存全局metadata(total_samples, env_metadata)
"""

import os
import json
import h5py
import argparse
from networkx import k_components
import numpy as np
from copy import deepcopy
from tqdm import tqdm

import imageio
import open3d as o3d

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.envs.env_base import EnvBase

import robomimic
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.vis_utils import depth_to_rgb
from robomimic.envs.env_base import EnvBase, EnvType
import robosuite

import colorsys
import random
import matplotlib.cm as cm
from PIL import Image

from robosuite.models.grippers import gripper_factory
from robosuite.robots import robot

from visualize_utils import *


DEFAULT_CAMERAS = {
    EnvType.ROBOSUITE_TYPE: ["agentview"],
    EnvType.IG_MOMART_TYPE: ["rgb"],
    EnvType.GYM_TYPE: ValueError("No camera names supported for gym type env!"),
}

RANGE_OF_WORKSPACE = {
    "lift": [[-0.36, -0.3, 0.810], [0.24, 0.3, 3.0]],
}

SUBTASK_CONTACT_FLAGS = {
    "lift": [False, True],
}


def get_frame_scene_from_obs(
    obs,
    time_step,
    camera_name,
):
    rgb_frame_scene = obs[camera_name + "_image"][time_step]
    d_frame_scene = obs[camera_name + "_depth"][time_step]

    return (
        rgb_frame_scene,
        d_frame_scene,
    )


def get_camera_info(
    env,
    camera_names=None,
    camera_height=84,
    camera_width=84,
):
    """
    Helper function to get camera intrinsics and extrinsics for cameras being used for observations.
    """

    # TODO: make this function more general than just robosuite environments
    assert EnvUtils.is_robosuite_env(env=env)

    if camera_names is None:
        return None

    camera_info = dict()
    for cam_name in camera_names:
        K = env.get_camera_intrinsic_matrix(
            camera_name=cam_name, camera_height=camera_height, camera_width=camera_width
        )
        R = env.get_camera_extrinsic_matrix(
            camera_name=cam_name
        )  # camera pose in world frame
        if "eye_in_hand" in cam_name:
            # convert extrinsic matrix to be relative to robot eef control frame
            assert cam_name.startswith("robot0")
            eef_site_name = env.base_env.robots[0].controller.eef_name
            eef_pos = np.array(
                env.base_env.sim.data.site_xpos[
                    env.base_env.sim.model.site_name2id(eef_site_name)
                ]
            )
            eef_rot = np.array(
                env.base_env.sim.data.site_xmat[
                    env.base_env.sim.model.site_name2id(eef_site_name)
                ].reshape([3, 3])
            )
            eef_pose = np.zeros((4, 4))  # eef pose in world frame
            eef_pose[:3, :3] = eef_rot
            eef_pose[:3, 3] = eef_pos
            eef_pose[3, 3] = 1.0
            eef_pose_inv = np.zeros((4, 4))
            eef_pose_inv[:3, :3] = eef_pose[:3, :3].T
            eef_pose_inv[:3, 3] = -eef_pose_inv[:3, :3].dot(eef_pose[:3, 3])
            eef_pose_inv[3, 3] = 1.0
            R = R.dot(eef_pose_inv)  # T_E^W * T_W^C = T_E^C
        camera_info[cam_name] = dict(
            intrinsics=K.tolist(),
            extrinsics=R.tolist(),
        )
    return camera_info


def extract_trajectory(
    env,
    initial_state,
    states,
    actions,
    done_mode,
    camera_names=None,
    camera_height=84,
    camera_width=84,
):
    """
    Helper function to extract observations, rewards, and dones along a trajectory using
    the simulator environment.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load to extract information
        actions (np.array): array of actions
        done_mode (int): how to write done signal. If 0, done is 1 whenever s' is a
            success state. If 1, done is 1 at the end of each trajectory.
            If 2, do both.
    """
    assert isinstance(env, EnvBase)
    assert states.shape[0] == actions.shape[0]

    # load the initial state
    env.reset()
    obs = env.reset_to(initial_state)

    # maybe add in intrinsics and extrinsics for all cameras
    camera_info = None
    is_robosuite_env = EnvUtils.is_robosuite_env(env=env)
    if is_robosuite_env:
        camera_info = get_camera_info(
            env=env,
            camera_names=camera_names,
            camera_height=camera_height,
            camera_width=camera_width,
        )

    # print(env.robots[0].base_pos)

    traj = dict(
        obs=[],
        next_obs=[],
        rewards=[],
        dones=[],
        actions=np.array(actions),
        states=np.array(states),
        initial_state_dict=initial_state,
    )
    traj_len = states.shape[0]
    # iteration variable @t is over "next obs" indices
    for t in range(1, traj_len + 1):

        # get next observation
        if t == traj_len:
            # play final action to get next observation for last timestep
            next_obs, _, _, _ = env.step(actions[t - 1])
        else:
            # reset to simulator state to get observation
            next_obs = env.reset_to({"states": states[t]})

        # infer reward signal
        # note: our tasks use reward r(s'), reward AFTER transition, so this is
        #       the reward for the current timestep
        r = env.get_reward()

        # infer done signal
        done = False
        if (done_mode == 1) or (done_mode == 2):
            # done = 1 at end of trajectory
            done = done or (t == traj_len)
        if (done_mode == 0) or (done_mode == 2):
            # done = 1 when s' is task success state
            done = done or env.is_success()["task"]
        done = int(done)

        # collect transition
        traj["obs"].append(obs)
        traj["next_obs"].append(next_obs)
        traj["rewards"].append(r)
        traj["dones"].append(done)

        # update for next iter
        obs = deepcopy(next_obs)

    # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
    traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return traj, camera_info


def dataset_states_to_obs(args):
    assert len(args.camera_names) > 0, "must specify camera names if using depth"

    # create environment to use for data processing
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=args.camera_names,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        reward_shaping=args.shaped,
        use_depth_obs=True,
    )

    # output file in same directory as input file
    exp_path = args.output_path
    os.makedirs(exp_path, exist_ok=True)

    # get env info
    env_metadata = env.serialize()
    print("==== Using environment with the following metadata ====")
    print(json.dumps(env_metadata, indent=4))
    print("")

    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)
    # print("is_robosuite_env: ", is_robosuite_env)

    # list of all demonstration episodes (sorted in increasing number order)
    hdf5_file = h5py.File(args.dataset, "r")
    # demos = list(hdf5_file["data"].keys())
    demos = [demo.decode('utf-8') for demo in hdf5_file["mask/picked_train"][()]]
    print("picked demos: ", demos)
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # create output file
    # demos: demo0, demo1, ...
    total_samples = 0
    for ind in tqdm(range(len(demos))):
        ep = demos[ind]  # episode name
        demo_path = exp_path + "/{}".format(ep)
        os.makedirs(demo_path, exist_ok=True)

        # prepare initial state to reload from
        states = hdf5_file["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        if is_robosuite_env:
            initial_state["model"] = hdf5_file["data/{}".format(ep)].attrs["model_file"]

        # extract obs, rewards, dones
        actions = hdf5_file["data/{}/actions".format(ep)][()]
        traj, camera_info = extract_trajectory(
            env=env,
            initial_state=initial_state,
            states=states,
            actions=actions,
            done_mode=args.done_mode,
            camera_names=args.camera_names,
            camera_height=args.camera_height,
            camera_width=args.camera_width,
        )

        # build a camera_info dictionary
        rgb_writer = [None for _ in args.camera_names]
        d_writer = [None for _ in args.camera_names]
        for i, cam_name in enumerate(args.camera_names):
            # create a video writer with imageio
            rgb_writer[i] = imageio.get_writer(
                demo_path + f"/rgb_{cam_name}.mp4", fps=20
            )
            d_writer[i] = imageio.get_writer(
                demo_path + f"/depth_{cam_name}.mp4", fps=20
            )

        robot_base_pos = env.env.sim.data.get_body_xpos("robot0_base")
        robot_base_quat = env.env.sim.data.get_body_xquat("robot0_base")
        #! mujoco: (w,x,y,z)

        ee_data_by_timestep = {}
        for k in range(len(traj["obs"]["robot0_eef_pos"])):

            frame_path = os.path.join(demo_path, f"{k}")
            os.makedirs(frame_path, exist_ok=True)

            gripper_left_finger = env.env.sim.data.get_body_xpos("gripper0_leftfinger")
            gripper_right_finger = env.env.sim.data.get_body_xpos(
                "gripper0_rightfinger"
            )
            gripper_eef = env.env.sim.data.get_body_xpos("gripper0_eef")

            ee_data = {
                "timestep": k,
                "robot0_ee_pos": traj["obs"]["robot0_eef_pos"][k].tolist(),
                "robot0_ee_quat": traj["obs"]["robot0_eef_quat"][k].tolist(),
                "robot0_gripper_qpos": traj["obs"]["robot0_gripper_qpos"][k].tolist(),
                "gripper_leftfinger": gripper_left_finger.tolist(),
                "gripper_rightfinger": gripper_right_finger.tolist(),
                "gripper_eef": gripper_eef.tolist(),
                "contact_flag": False,
                "subtask_index": 0,
            }
            ee_data_by_timestep[k] = ee_data

            # write to image
            for i, cam_name in enumerate(args.camera_names):
                (
                    rgb_frame_scene,
                    d_frame_scene,
                ) = get_frame_scene_from_obs(
                    traj["obs"],
                    k,
                    cam_name,
                )

                rgb_path = os.path.join(frame_path, f"{k}_rgb_{cam_name}.png")
                imageio.imwrite(rgb_path, rgb_frame_scene)
                d_path = os.path.join(frame_path, f"{k}_depth_mm_{cam_name}.npy")
                depth_map_meters = d_frame_scene[..., 0]
                depth_map_mm = (depth_map_meters * 1000).astype(np.uint16)
                np.save(d_path, depth_map_mm)
                print(f"Saved image for frame #{k}")

                rgb_writer[i].append_data(rgb_frame_scene)
                d_frame_scaled = (
                    (d_frame_scene - d_frame_scene.min())
                    / (d_frame_scene.max() - d_frame_scene.min())
                    * 255
                )
                d_scene = d_frame_scaled.astype(np.uint8)
                d_writer[i].append_data(d_scene)

        for i, cam_name in enumerate(args.camera_names):
            rgb_writer[i].close()
            d_writer[i].close()

        # ee_data_by_timestep
        with open(demo_path + "/ee_data.json", "w") as f:
            json.dump(ee_data_by_timestep, f, indent=4)

        # episode metadata
        num_samples = traj["actions"].shape[0]  # number of transitions in this episode
        ep_data = {
            "episode": ep,
            "num_samples": num_samples,
            "trajectory_length": len(traj["obs"]["robot0_eef_pos"]),
            "robot_base_pos": robot_base_pos.tolist(),
            "robot_base_quat": robot_base_quat.tolist(),
            "camera_names": args.camera_names,
            "camera_info": camera_info,
            "camera_height": args.camera_height,
            "camera_width": args.camera_width,
            "range_of_workspace": RANGE_OF_WORKSPACE[args.task],
            "end_of_subtasks": [],
            "subtask_contact_flags": SUBTASK_CONTACT_FLAGS[args.task],
        }
        with open(demo_path + "/episode_metadata.json", "a") as f:
            json.dump(ep_data, f, indent=4)
            f.write("\n")

        total_samples += traj["actions"].shape[0]

    # global metadata
    global_data = {
        "total_samples": total_samples,
        "env_metadata": env_metadata,
    }
    with open(exp_path + "/global_metadata.json", "a") as f:
        json.dump(global_data, f, indent=4)
        f.write("\n")

    print("Wrote {} trajectories to {}".format(len(demos), exp_path))
    hdf5_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to input hdf5 dataset. There must be a key 'picked_train' in the mask",
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="which task to extract data for. Must be a key in the dataset.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="name of output dir, full path",
    )

    # flag for reward shaping
    parser.add_argument(
        "--shaped",
        action="store_true",
        help="(optional) use shaped rewards",
    )

    # camera names to use for observations
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs="+",
        default=[],
        help="(optional) camera name(s) to use for image observations. Leave out to not use image observations.",
    )

    parser.add_argument(
        "--camera_height",
        type=int,
        default=84,
        help="(optional) height of image observations",
    )

    parser.add_argument(
        "--camera_width",
        type=int,
        default=84,
        help="(optional) width of image observations",
    )

    # specifies how the "done" signal is written. If "0", then the "done" signal is 1 wherever
    # the transition (s, a, s') has s' in a task completion state. If "1", the "done" signal
    # is one at the end of every trajectory. If "2", the "done" signal is 1 at task completion
    # states for successful trajectories and 1 at the end of all trajectories.
    parser.add_argument(
        "--done_mode",
        type=int,
        default=0,
        help="how to write done signal. If 0, done is 1 whenever s' is a success state.\
            If 1, done is 1 at the end of each trajectory. If 2, both.",
    )

    args = parser.parse_args()
    dataset_states_to_obs(args)
