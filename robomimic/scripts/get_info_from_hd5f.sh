#!/bin/bash
###
 # @Author: Shuying Deng
 # @Date: 2024-03-23 15:40:37
 # @LastEditTime: 2024-03-25 23:52:12
 # @UpdateRecord: 
 # @FilePath: /demo_generate/DemoGen/get_info.sh
 # @Description: 用于从raw hdf5数据集中提取可视化的观测数据的脚本
### 

BASE_DATASET_DIR="../../datasets"
echo "Using base dataset directory: $BASE_DATASET_DIR"

###########  world frame, rgbd in png/npy, without seg  ############

python dataset_to_info_world_frame.py --dataset $BASE_DATASET_DIR/lift/ph/demo_v141.hdf5 \
--done_mode 2 \
--camera_names agentview --camera_height 84 --camera_width 84 \
--exp_name test --task lift \


###########   world frame, rgbd in png/npy  ############

# python dataset_to_low_dim_info_world.py --dataset $BASE_DATASET_DIR/"$2"/ph/demo_v141.hdf5 \
# --done_mode 2 \
# --camera_names agentview --camera_height 84 --camera_width 84 \
# --camera_segmentations instance \
# --exp_name "$1" --task "$2" \


###########  robot base frame, pcd in first and last frame  ############

# python dataset_to_all_info.py --dataset $BASE_DATASET_DIR/"$2"/ph/demo_v141.hdf5 \
# --done_mode 2 \
# --camera_names agentview --camera_height 512 --camera_width 512 \
# --camera_segmentations instance \
# --exp_name "$1" --task "$2" \
# --first_and_last_pcd


###########  robot base frame, gray pcd ############

# python dataset_to_all_info.py --dataset $BASE_DATASET_DIR/"$2"/ph/demo_v141.hdf5 \
# --done_mode 2 \
# --camera_names agentview --camera_height 512 --camera_width 512 \
# --camera_segmentations instance \
# --exp_name "$1" --task "$2" \
# --gray_pointcloud \


###########   robot base frame, rgbd in png/npy  ############

# python dataset_to_low_dim_info.py --dataset $BASE_DATASET_DIR/"$2"/ph/demo_v141.hdf5 \
# --done_mode 2 \
# --camera_names agentview --camera_height 84 --camera_width 84 \
# --camera_segmentations instance \
# --exp_name "$1" --task "$2" \