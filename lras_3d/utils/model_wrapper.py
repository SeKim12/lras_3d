GCLOUD_BUCKET_NAME = "lras_3d"
GCLOUD_MODELS_PATH = "lras_3d/models"
GCLOUD_URL_NAME = "https://storage.googleapis.com/lras_3d"

import gc
import torch
import os
import inspect
from typing import Tuple
import requests
import tqdm
import importlib
from dataclasses import dataclass
import subprocess
from google.cloud import storage

"""
Wrapper class for pytorch models

It provides a few core functionality components
- Model saving to google clour
- Model loading from google cloud
- Counting the number of parameters in the model
- Calculating the power usage of the model (in watts-hours)
"""
class WrappedModel(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

    def save(self, elements_to_save: dict, path: str, gcloud: bool = True, bucket_name: str = GCLOUD_BUCKET_NAME):
        """
        Save the model to the given path and upload it to google cloud
        
        Args:
        elements_to_save: dict
            Dictionary containing the elements to save
        path: str
            Path to save the model
        
        If model and cfg are not in elements_to_save, they will be added by default
        """

        print("Entering saving loop", flush=True)
        if elements_to_save["args"].device != "xla":
            # Add the model and the configuration to the elements to save
            if 'model' not in elements_to_save and 'weights' not in elements_to_save:
                elements_to_save['model'] = self.state_dict()
            if 'cfg' not in elements_to_save:
                    elements_to_save['cfg'] = cfg_to_dict(self.config)
            # Save elements to the given path (unless the device is xla in which case we already saved it)
            torch.save(elements_to_save, path)

        print(f"💾 Model saved to {path}", flush=True)
        if gcloud:
            from google.cloud import storage
            # Initialize the storage client
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            # Grab the directory and model name from local save path
            model_dir_and_name = "/".join(path.split("/")[-2:])
            blob = bucket.blob(f'{GCLOUD_MODELS_PATH}/{model_dir_and_name}')
            # Upload the model to google cloud
            blob.upload_from_filename(path)
            # Make the model public
            # blob.make_public()
            print(f"☁️  Model uploaded to {bucket_name} bucket", flush=True)

    def get_num_params(self):
        """
        Count the number of parameters in the model
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def configure_optimizers(self, weight_decay: float, learning_rate: float, betas: Tuple, device_type: str):
        """
        Configure the optimizer for the model

        Args:
        weight_decay: float
            Weight decay for the optimizer
        learning_rate: float
            Learning rate for the optimizer
        betas: tuple
            Betas for the optimizer
        device_type: str
            Type of the device (cpu or cuda)

        Returns:
        optimizer: torch.optim.Optimizer
            Optimizer for the model
        """
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer


"""
Model Factory for loading a checkpoint from gcloud, then initializing the model and the configuration
"""
class ModelFactory:

    def __init__(self, bucket_name: str = GCLOUD_BUCKET_NAME):
        self.bucket_name = bucket_name
    
    def get_catalog(self):
        """
        Get the list of models in the bucket
        """
        # Initialize the storage client
        storage_client = storage.Client()
        bucket = storage_client.bucket(self.bucket_name)
        # Get the list of blobs in the bucket
        blobs = bucket.list_blobs()
        return [blob.name for blob in blobs]

    def load_model(self, model_name: str, force_download=False, return_ckpt=False, return_optimizer=False):
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

        # Find root of git repo, use a directory inside of it as the checkpoint path
        checkpoint_path = os.path.join(self.find_git_repo_path(), 'checkpoints')
        os.makedirs(checkpoint_path, exist_ok=True)

        # If force_download is true or the model has not yet been downloaded, grab it from gcloud
        if force_download or not os.path.exists(os.path.join(checkpoint_path, model_name)):
            # Construct gcloud url
            gcloud_url = f"{GCLOUD_URL_NAME}/{GCLOUD_MODELS_PATH}/{model_name}"
            # Download the model from google cloud using requests (with tqdm timer)
            response = requests.get(gcloud_url, stream=True)
            total_size_in_bytes= int(response.headers.get('content-length', 0))
            block_size = 1024 # 1 Kibibyte
            progress_bar = tqdm.tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
            # Ensure checkpoint path exists and download the model
            os.makedirs(os.path.join(checkpoint_path, "/".join(model_name.split("/")[:-1])), exist_ok=True)
            with open(os.path.join(checkpoint_path, model_name), 'wb') as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)

        # Initialize the model and the configuration from the checkpoint
        print(f"Loading model from {os.path.join(checkpoint_path, model_name)}", flush=True)
        ckpt = torch.load(os.path.join(checkpoint_path, model_name), map_location="cpu", weights_only=False)

        # NOTE: Temporary fix for an issue where optimizer causes OOM issue
        if 'optimizer' in ckpt and not return_optimizer: 
            del ckpt['optimizer']
            gc.collect()
        
        ## HACK: change config on-the-fly
        if "quantizer" not in model_name:
            ckpt['cfg']['config_class'] = "lras_3d.lras_3d.config.LRAS7B_Patch8_RGBConfig"
            ckpt['cfg']['model_type'] = "lras_3d.utils.model_wrapper.WrappedModel"
            ckpt['cfg']['model_class'] = "lras_3d.lras_3d.model.LRAS"
        else:
            ckpt['cfg']['config_class'] = "lras_3d.local_patch_quantizer.config.LPQConfig"
            ckpt['cfg']['model_type'] = "lras_3d.utils.model_wrapper.WrappedModel"
            ckpt['cfg']['model_class'] = "lras_3d.local_patch_quantizer.model.LPQ"

        # Get the config from the checkpoint
        config = dict_to_cfg(ckpt['cfg'])

        # Initialize the model from the configuration
        # Separate the module path and class name
        module_path, class_name = config.model_class.rsplit('.', 1)
        # Import the module from the given path
        module = importlib.import_module(module_path)
        # Get the class from the module by its name
        model_class = getattr(module, class_name)
        # Initialize the model from the model class and the configuration
        model = model_class(config)

        # Load the model from the checkpoint
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt['weights']
        # Remove the prefix "_orig_mod." from the state_dict keys
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)

        num_params_millions = model.get_num_params() / 1e6  # Convert to millions
        rounded_params = round(num_params_millions, 1)  # Round to 1 decimal place

        # Determine the longest line to dynamically set the width of the horizontal dashes
        max_line_length = max(
            len(f"📂 Model loaded from:"),
            len(f"{model_name}"),
            len(f"{checkpoint_path}"),
            len(f"🔢 {rounded_params}M parameters in the model"),
        ) + 4  # Add some padding

        # Print the model information overview with dynamic width
        print(f"|{'-' * (max_line_length - 2)}|")
        print(f"|{'✨ Model Information Overview ✨'.center(max_line_length - 4)}|")
        print(f"|{'-' * (max_line_length - 2)}|")
        print(f"| {'📦 Model Name:'.ljust(max_line_length - 4)}|")
        print(f"| {model_name:<{max_line_length - 3}}|")
        print(f"|{'-' * (max_line_length - 2)}|")
        print(f"| 📂 Model loaded from: {'|'.rjust(max_line_length - 24)}")
        print(f"| {checkpoint_path:<{max_line_length - 3}}|")
        print(f"|{'-' * (max_line_length - 2)}|")
        print(f"| 🔢 {rounded_params}M parameters in the model{'|'.rjust(max_line_length - len(str(rounded_params)) - 30)}")
        print(f"|{'-' * (max_line_length - 2)}|")

        if return_ckpt:
            return model, ckpt

        return model
    
    def load_model_from_config(self, config):
        """
        Load the model from the given configuration
        
        Args:
        config: Config or dict
            Configuration to load the model from
        
        Returns:
        model: Model
            Model initialized from the configuration
        """
        # if config is a dictionary, convert it to a config
        if isinstance(config, dict):
            config = dict_to_cfg(config)
        
        config.model_type = 'lras_3d.lras_3d.model.LRAS'
        # Separate the module path and class name
        module_path, class_name = config.model_type.rsplit('.', 1)
        
        # Import the module from the given path
        module = importlib.import_module(module_path)
        
        # Get the class from the module by its name
        model_class = getattr(module, class_name)
        
        # Initialize the model from the model class and the configuration
        model = model_class(config)
        
        return model
    
    def load_model_from_checkpoint(self, checkpoint_path):
        """
        Load the model from the given checkpoint path
        
        Args:
        checkpoint_path: str
            Path to the checkpoint to load the model from
        
        Returns:
        model: Model
            Model initialized from the checkpoint
        """
        # Load the checkpoint
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        
        # Get the config from the checkpoint
        config = dict_to_cfg(ckpt['cfg'])
        
        # Initialize the model from the configuration
        model = self.load_model_from_config(config)
        
        # Load the model from the checkpoint
        model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt['weights'])
        
        return model

    def find_git_repo_path(self, start_path=None):
        """
        Traverse up from the start_path until a directory containing a .git folder is found.
        
        Args:
        - start_path: The path to start the search from. If None, the current directory is used.
        
        Returns:
        - The path of the directory where the .git folder is found, or None if not found.
        """
        if start_path is None:
            # Use the current directory if no start path is given
            start_path = os.getcwd()
        current_path = start_path
        
        while True:
            if os.path.isdir(os.path.join(current_path, '.git')):
                # .git directory found, return the current path
                return current_path
            parent_path = os.path.dirname(current_path)
            
            if parent_path == current_path:
                # Root directory reached without finding .git
                return None
            
            # Move up to the parent directory
            current_path = parent_path

    def list_checkpoints(self, model_name: str):
        breakpoint()
        try:
            # Create a client
            print(f'Listing checkpoints in {model_name}\nIf there is a credential issue, you need to run: `gcloud auth application-default login`')
            client = storage.Client()

            # Get the bucket
            bucket = client.get_bucket(GCLOUD_BUCKET_NAME)

            # List all the blobs in the specified folder
            blobs = bucket.list_blobs(prefix=f"{GCLOUD_MODELS_PATH}/{model_name}/")

            # Print the file names
            checkpoints = []
            for blob in blobs:
                checkpoint_name = blob.name.split('/')[-1]
                if '.pt' in checkpoint_name:
                    checkpoints.append(f"{model_name}/{checkpoint_name}")

            print(f'{len(checkpoints)} checkpoints found in {model_name}')
            return checkpoints


        except Exception as e:
            print(f"Error listing folders: {e}")
            print("Trying to use the local checkpoints")
            import glob
            github_repo_path = self.find_git_repo_path()
            checkpoints = glob.glob(os.path.join(github_repo_path, f"checkpoints/{model_name}/*.pt"))
            # if there is no checkpoints, return an empty list
            if len(checkpoints) == 0:
                print("No checkpoints found in the local directory")
                return []
            # only last two folders
            checkpoints = [os.path.join(*checkpoint.split('/')[-2:]) for checkpoint in checkpoints]
            return checkpoints

"""
Default parent class for all config classes
"""

@dataclass
class BaseConfig():
    model_type: str = "lras_3d.utils.model_wrapper.WrappedModel"


"""
Function that converts a config to a dictionary
"""
def cfg_to_dict(cfg):
    """
    Convert the config to a dictionary

    Makes a dictionary for every field in the config and returns it
    Adds an extra field for class of the config
    If any of the fields are also a config, it will call to_dict on that config
    and save the it as a dictionary in the main dictionary
    """
    full_class_name = f"{cfg.__class__.__module__}.{cfg.__class__.__name__}"
    cfg_dict = {
        "config_class": full_class_name
    }
    for field in cfg.__dataclass_fields__:
        value = getattr(cfg, field)
        if isinstance(value, BaseConfig):
            cfg_dict[field] = cfg_to_dict(value)
        else:
            cfg_dict[field] = value
    return cfg_dict


"""
Function that converts a dictionary to a config
"""
def dict_to_cfg(cfg_dict):
    """
    Convert the dictionary to a config

    Makes a config for every field in the dictionary and returns it
    If any of the fields are also a dictionary, it will call from_dict on that config
    and save the it as a config in the main config
    """

    # Separate the module path and class name
    module_path, cfg_name = cfg_dict["config_class"].rsplit('.', 1)
    # Import the module from the given path
    module = importlib.import_module(module_path)
    # Get the class from the module by its name
    cfg_class = getattr(module, cfg_name)
    # Initialize the model from the model class and the configuration
    cfg = cfg_class()

    for field in cfg.__dataclass_fields__:
        if field in cfg_dict:
            value = cfg_dict[field]
            if isinstance(value, dict):
                setattr(cfg, field, dict_to_cfg(value))
            else:
                setattr(cfg, field, value)
    return cfg
