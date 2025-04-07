from setuptools import setup, find_packages


setup(
    name="lras_3d",
    version="0.1",
    packages=find_packages(),
    description="lras_3d: 3D Understanding Through Local Random Access Sequence Modeling",
    author="Stanford NeuroAI Lab",
    install_requires=[
        'numpy==1.24.0',
        'torch',
        'scipy',
        'tqdm',
        'wandb',
        'einops',
        'matplotlib',
        'h5py',
        'torchvision',
        'future',
        'opencv-python',
        'decord',
        'pandas',
        'matplotlib',
        'moviepy',
        'scikit-image',
        'scikit-learn',
        'vector_quantize_pytorch',
        'google-cloud-storage',
        'gradio',
        'lpips',
        'patchy',
        'timm'
    ],
)
