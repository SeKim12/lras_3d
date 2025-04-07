import torch
import os
import requests
import tqdm
import importlib
from dataclasses import fields


GCLOUD_BUCKET_NAME = "stanford_neuroai_models"
GCLOUD_URL_NAME = "https://storage.googleapis.com/stanford_neuroai_models"
CACHE_PATH = f"{os.getenv('CACHE')}/stanford_neuroai_models" if os.getenv('CACHE') is not None else ".cache/stanford_neuroai_models"


_model_catalogue ={
    "lras_3d_patch_base": {
        "path": "lras_3d/lras_3d_patch_base.pt",
    },
}

"""
Model Factory for loading a checkpoint from gcloud, then initializing the model and the configuration
"""
class ModelFactory:

    def __init__(self, bucket_name: str = GCLOUD_BUCKET_NAME):
        self.bucket_name = bucket_name
    
    def get_catalog(self):
        """
        Get the list of available models
        """
        # Initialize the storage client
        return _model_catalogue.keys()

    def load_model(self, model_name: str, force_download=False):
        """
        Load the model given the name
        
        Args:
        model_name: str
            Name of the model to load
        force_download: bool (optional)
            Whether to force the download of the freshest weights from gcloud
        
        Returns:
        model: torch.nn.Module
            Model initialized from the checkpoint
        """

        # Find cache dir, use a directory inside of it as the checkpoint path
        checkpoint_path = os.path.join(CACHE_PATH, _model_catalogue[model_name]["path"])
        # Make checkpoint directory if it does not exist
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        # Construct gcloud url
        gcloud_url = os.path.join(GCLOUD_URL_NAME, _model_catalogue[model_name]['path'])
        # Download the model from google cloud using requests (with tqdm timer)
        response = requests.get(gcloud_url, stream=True)
        total_size_in_bytes= int(response.headers.get('content-length', 0))
        block_size = 1024 # 1 Kibibyte
        progress_bar = tqdm.tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        
        # If force_download is true or the model has not yet been downloaded, grab it from gcloud

        if force_download or not os.path.exists(checkpoint_path):
            print(f"Saving model to cache: {CACHE_PATH}")
            with open(checkpoint_path, 'wb') as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)

        # Initialize the model and the configuration from the checkpoint
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        config = ckpt['config']

        # Initialize the model from the configuration

        # Seperate the cfg_class from the model name
        cfg_module_path, cfg_class_name = config['cfg_class'].rsplit('.', 1)
        # Import the module from the given path
        cfg_module = importlib.import_module(cfg_module_path)
        # Get the class from the module by its name
        cfg_class = getattr(cfg_module, cfg_class_name)
        # Get all of the field names from the configuration class
        cfg_class_keys = [f.name for f in fields(cfg_class)]
        # Initialize the configuration from the configuration class and the configuration
        cfg = cfg_class(**{k: v for k, v in config.items() if k in cfg_class_keys})

        # Separate the module path and class name
        module_path, class_name = config['model_class'].rsplit('.', 1)
        # Import the module from the given path
        module = importlib.import_module(module_path)
        # Get the class from the module by its name
        model_class = getattr(module, class_name)
        # Initialize the model from the model class and the configuration
        model = model_class(cfg)

        # Load the model from the checkpoint
        model.load_state_dict(ckpt['model'], strict=True)

        return model
