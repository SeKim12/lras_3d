# Installation

```bash
conda create -n lras3d python=3.9 -y
conda activate lras3d
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118

git clone https://github.com/neuroailab/lras_3d.git
cd lras_3d
pip install -e .

# SAM
pip install git+https://github.com/facebookresearch/segment-anything.git

# Depth Anything v2 (metric)
pip install -r external/depth_anything_v2/requirements.txt

# Pytorch3d
wget https://anaconda.org/pytorch3d/pytorch3d/0.7.8/download/linux-64/pytorch3d-0.7.8-py39_cu118_pyt231.tar.bz2
conda install ./pytorch3d-0.7.8-py39_cu118_pyt231.tar.bz2
rm pytorch3d-0.7.8-py39_cu118_pyt231.tar.bz2

```
