#!/bin/bash

BASE_DATASET_DIR="../../datasets"
echo "Using base dataset directory: $BASE_DATASET_DIR"

###################### img+depth ######################

# lift - ph
python dataset_states_to_obs.py --dataset $BASE_DATASET_DIR/lift/ph/demo_v141.hdf5 \
--output_name img_depth_200_84x84_v141.hdf5 --done_mode 2 \
--camera_names agentview --camera_height 84 --camera_width 84 --depth

python dataset_states_to_obs.py --dataset $BASE_DATASET_DIR/lift/ph/demo_v141.hdf5 \
--output_name img_depth_200_256x256_v141.hdf5 --done_mode 2 \
--camera_names agentview --camera_height 256 --camera_width 256 --depth

# # can - ph
# python dataset_states_to_obs.py --dataset $BASE_DATASET_DIR/can/ph/demo_v141.hdf5 \
# --output_name img_depth_v141.hdf5 --done_mode 2 \
# --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 --depth

# # square - ph
# python dataset_states_to_obs.py --dataset $BASE_DATASET_DIR/square/ph/demo_v141.hdf5 \
# --output_name img_depth_v141.hdf5 --done_mode 2 \
# --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 --depth

# # transport - ph
# python dataset_states_to_obs.py --dataset $BASE_DATASET_DIR/transport/ph/demo_v141.hdf5 \
# --output_name img_depth_v141.hdf5 --done_mode 2 \
# --camera_names shouldercamera0 shouldercamera1 robot0_eye_in_hand robot1_eye_in_hand \
# --camera_height 84 --camera_width 84 --depth

# # tool hang - ph
# python dataset_states_to_obs.py --dataset $BASE_DATASET_DIR/tool_hang/ph/demo_v141.hdf5 \
# --output_name img_depth_v141.hdf5 --done_mode 2 \
# --camera_names sideview robot0_eye_in_hand --camera_height 240 --camera_width 240 --depth



###################### img+depth+seg ######################

# python dataset_states_to_obs_all.py --dataset $BASE_DATASET_DIR/lift/ph/demo_v141.hdf5 \
# --output_name all_v141.hdf5 --done_mode 2 \
# --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 \
# --camera_segmentations instance

# python dataset_states_to_obs_all.py --dataset $BASE_DATASET_DIR/can/ph/demo_v141.hdf5 \
# --output_name all_v141.hdf5 --done_mode 2 \
# --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 \
# --camera_segmentations instance

# python dataset_states_to_obs_all.py --dataset $BASE_DATASET_DIR/square/ph/demo_v141.hdf5 \
# --output_name all_v141.hdf5 --done_mode 2 \
# --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 \
# --camera_segmentations instance

# python dataset_states_to_obs_all.py --dataset $BASE_DATASET_DIR/transport/ph/demo_v141.hdf5 \
# --output_name all_v141.hdf5 --done_mode 2 \
# --camera_names shouldercamera0 shouldercamera1 robot0_eye_in_hand robot1_eye_in_hand \
# --camera_height 84 --camera_width 84 --camera_segmentations instance

# python dataset_states_to_obs_all.py --dataset $BASE_DATASET_DIR/tool_hang/ph/demo_v141.hdf5 \
# --output_name all_v141.hdf5 --done_mode 2 \
# --camera_names sideview robot0_eye_in_hand --camera_height 240 --camera_width 240 \
# --camera_segmentations instance