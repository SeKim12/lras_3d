#!/bin/bash

# Base URL for downloading files
BASE_URL="https://storage.googleapis.com/lras_3d/lras_3d/models/checkpoints"

# Create directories
mkdir -p checkpoints/depth_anything_v2
mkdir -p checkpoints/flow_predictor
mkdir -p checkpoints/flow_quantizer
mkdir -p checkpoints/rgb_predictor
mkdir -p checkpoints/rgb_quantizer
mkdir -p checkpoints/sam

# Download files
wget ${BASE_URL}/depth_anything_v2/depth_anything_v2_metric_hypersim_vitl.pth -P checkpoints/depth_anything_v2/
wget ${BASE_URL}/depth_anything_v2/depth_anything_v2_metric_vkitti_vitl.pth -P checkpoints/depth_anything_v2/
wget ${BASE_URL}/flow_predictor/flow_predictor.pt -P checkpoints/flow_predictor/
wget ${BASE_URL}/flow_quantizer/flow_quantizer.pt -P checkpoints/flow_quantizer/
wget ${BASE_URL}/rgb_predictor/rgb_predictor.pt -P checkpoints/rgb_predictor/
wget ${BASE_URL}/rgb_quantizer/rgb_quantizer.pt -P checkpoints/rgb_quantizer/
wget ${BASE_URL}/sam/sam_vit_h_4b8939.pth -P checkpoints/sam/
