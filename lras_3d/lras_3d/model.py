"""
Pixel-wise Causal Counterfactual World Model (LRAS) implmentation

The model utilizes a GPT-style
"""

import math
import inspect
from typing import Tuple, Union, List
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import tqdm

from lras_3d.utils.model_wrapper import WrappedModel
from lras_3d.utils.modeling import LayerNorm, Block
from lras_3d.utils.image_processing import patchify, unpatchify, unpatchify_logits, convert_from_16bit_color



class LRAS(WrappedModel):

    def __init__(self, config):
        super().__init__(config)
        print("using config:", config, flush=True)
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            token_embedding = nn.Embedding(config.vocab_size, config.n_embd),
            positional_embedding = nn.Embedding(config.block_size, config.n_embd),
            # positional_embedding = nn.Embedding(3*4096, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # self.lm_head = nn.Linear(config.n_embd, 512, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate

        # self.transformer.token_embedding.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
        self.unsharded_param_count = self.get_num_params()

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.positional_embedding.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(
            self, 
            seq: torch.Tensor, 
            pos: torch.Tensor = None, 
            tgt: torch.Tensor = None, 
            mask: torch.Tensor = None,
            exotic_mask: str = None,
        ) -> torch.Tensor:
        """
        Forward pass of the model

        Parameters:
            seq (torch.Tensor) of size b, t: The input sequence
            pos (torch.Tensor) of size b, t: The positional indices of the sequence
            tgt (torch.Tensor) of size b, t_tgt: The target sequence
            mask (torch.Tensor) of size b, t, t: The mask of the sequence
        
        Returns:
            torch.Tensor: The logits of the model. Size b, t if tgt is None, else b, t_tgt
        """

        # grab device to perform operations on
        device = seq.device
        # grab dimensions
        b, t = seq.size()
        if tgt is not None:
            b, t_tgt = tgt.size()

        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        assert t_tgt <= t, \
            f"Target seqeunce length {t_tgt} must be shorter than or equal to sequence length {t}"

        # create a tensor of position indices, if not provided
        if pos is None:
            pos = torch.arange(t, device=device).unsqueeze(0).expand(b, -1)

        # forward the GPT model itself
        tok_emb = self.transformer.token_embedding(seq) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.positional_embedding(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)

        # if tgt is not none, compute the logits for the entire sequence
        if tgt is None:
            logits = self.lm_head(x)
            return logits, None
        
        # if tgt is not none, compute the logits and the loss for the target sequence
        logits = self.lm_head(x[:, -tgt.size(1):])

        # set all target tokens above 65535 to -1 so they are not included in the loss
        # we do this to ignore the prediction of the patch indexes since they are in random order
        # tgt[((65535+512) > tgt) & (tgt > 65535)] = -1
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-1)
        return logits, loss


    def sample_logits(self, logits: torch.FloatTensor, temp: float = 1.0, 
                      top_k: int = 1000, top_p: float = 0.9, 
                      blacklist: List[int] = None) -> torch.LongTensor:
        """
        Samples an integer from the distribution of logits

        Parameters:
            logits (torch.FloatTensor): The logits of the distribution
            temp (float): The temperature of the sampling, if 0.0, then argmax is used
            top_k (int): The number of top k tokens to consider during sampling
            top_p (float): The cumulative probability threshold for nucleus (top-p) sampling
            blacklist (List[int]): The list of tokens to blacklist during sampling
        Returns:
            torch.LongTensor: The sampled integer
        """

        # If temperature is 0.0, use argmax
        if temp == 0.0:
            return torch.argmax(logits, dim=-1)
        
        # Apply temperature
        logits = logits / temp

        # Apply top-k filtering if specified
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[..., [-1]]] = -float('Inf')

        # Apply top-p (nucleus) filtering if specified
        if top_p is not None:
            # Sort the logits in descending order
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            # Compute the sorted softmax probabilities
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            # Compute the cumulative probabilities
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            # Create a mask for tokens to remove
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the mask right to keep at least one token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            # Scatter the mask back to the original indices
            indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
            logits[indices_to_remove] = -float('Inf')
        
        # Compute softmax probabilities
        probs = F.softmax(logits, dim=-1)
        # Flatten probabilities to (batch_size * sequence_length, vocab_size)
        flat_probs = probs.view(-1, probs.size(-1))
        # Sample from the distribution
        sampled = torch.multinomial(flat_probs, num_samples=1)
        # Reshape to original shape except for the last dimension
        sampled = sampled.view(*logits.shape[:-1])
        return sampled
    
    def encode_kv_cache(self, seq: torch.Tensor, pos: torch.Tensor, return_last_logits: bool = False,
                        k_cache: torch.Tensor = None, v_cache: torch.Tensor = None):
        """
        Encode the key and value cache for the given sequence

        Parameters:
            seq (torch.Tensor) of size b, t: The input sequence
        
        Returns:
            Tuple of torch.Tensor: The key and value cache of the sequence
        """

        # grab device to perform operations on
        device = seq.device
        # grab dimensions
        b, t = seq.size()

        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # create a tensor of position indices
        if pos is None:
            pos = torch.arange(t, device=device).unsqueeze(0).expand(b, -1)

        # forward the GPT model itself
        tok_emb = self.transformer.token_embedding(seq)
        pos_emb = self.transformer.positional_embedding(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        k_list = []
        v_list = []
        for block_idx, block in enumerate(self.transformer.h):
            if k_cache is not None and v_cache is not None:
                x, k, v = block(x, k_cache=k_cache[block_idx], v_cache=v_cache[block_idx], return_kv=True)
            else:
                x, k, v = block(x, return_kv=True)
            k_list.append(k)
            v_list.append(v)
        # k_cache and v_cache have shape (n_layer, b, n_head, t, n_embd//n_head)
        k_cache = torch.stack(k_list, dim=0)
        v_cache = torch.stack(v_list, dim=0)

        if return_last_logits:
            x = self.transformer.ln_f(x)
            logits = self.lm_head(x)
            logits = logits[:, [-1]]
            return k_cache, v_cache, logits
        else:
            return k_cache, v_cache
    
    @torch.no_grad()
    def rollout_kv_cache(
        self, 
        seq: torch.Tensor, 
        k_cache: torch.Tensor = None, 
        v_cache: torch.Tensor = None, 
        num_new_tokens: int = 4148, 
        temperature: float = 1.0, 
        return_cache: bool = False,
        patch_indexes: torch.Tensor = None, 
        pos: torch.Tensor = None,
        top_k = None, 
        top_p = None, 
        causal_mask_length: int = None,
        n_tokens_per_patch: int = None,
        sample_range: Tuple[int, int] = None,
        sampling_blacklist: List[List[int]] = None,
        num_unmasked_tokens: int = 0,
        remaining_seq=None,
    ) -> torch.Tensor:
        """
        Rollout the key and value cache for the given sequence

        Parameters:
            seq (torch.Tensor) of size b, t: 
                The input sequence
            k_cache (torch.Tensor) of size n_layer, b, n_head, t, n_embd//n_head: 
                The key cache
            v_cache (torch.Tensor) of size n_layer, b, n_head, t, n_embd//n_head: 
                The value cache
            num_new_tokens (int): 
                The number of new tokens to rollout
            temperature (float): 
                The temperature of the sampling
            return_cache (bool):
                Whether to return the key and value cache
            patch_indexes (torch.Tensor) of size b, t:
                The indexes of the patches to rollout from
            pos (torch.Tensor) of size b, t:
                The positional indices of the sequence
            top_k (int):
                The number of top k tokens to consider during sampling
            top_p (float):
                The cumulative probability threshold for nucleus (top-p) sampling
        
        Returns:
            Tuple of torch.Tensor: 
                The key and value cache of the sequence
        """


        sample_range = None

        # if temp, top_k, and top_p are not lists, make them lists of length num_new_tokens
        if not isinstance(temperature, list):
            temperature = [temperature] * num_new_tokens
        if not isinstance(top_k, list):
            top_k = [top_k] * num_new_tokens
        if not isinstance(top_p, list):
            top_p = [top_p] * num_new_tokens

        # grab device to perform operations on

        # breakpoint()

        device = seq.device
        # grab dimensions
        b, t = seq.size()
        # grab number of tokens per patch (patch_size^2 + 1 for the index token)
        if n_tokens_per_patch is None:
            n_tokens_per_patch = self.config.patch_size**2 + 1

        all_logits = []

        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        # create a tensor of position indices if not provided
        if pos is None:
            pos = torch.arange(8710, device=device).unsqueeze(0).expand(b, -1)

        for i in tqdm.tqdm(range(num_new_tokens), desc='Rolling out sequence'):

            # create a tensor of position indices
            tok_pos = pos[:, :seq.size(1)]

            # forward the GPT model itself
            if i == 0: # if we are on the first pass push the entire sequence through
                tok_emb = self.transformer.token_embedding(seq)
                pos_emb = self.transformer.positional_embedding(tok_pos)
            else: # else just trim the last token and push it through
                tok_emb = self.transformer.token_embedding(seq[:, [-1]])
                pos_emb = self.transformer.positional_embedding(tok_pos[:, [-1]])
            x = self.transformer.drop(tok_emb + pos_emb)
            k_list = []
            v_list = []
            for block_idx, block in enumerate(self.transformer.h):
                # if k and v cache are passed in, use them
                if k_cache is not None and v_cache is not None:
                    x, k, v = block(x, k_cache=k_cache[block_idx], v_cache=v_cache[block_idx])
                # else if if k and v cache are not passed in compute them from the sequence
                else:
                    mask = None
                    if causal_mask_length is not None:
                        mask = torch.zeros(x.shape[1], x.shape[1], device=x.device).bool()
                        # first t - causal_mask_length tokens cannot attend to the last causal_mask_length tokens
                        mask[-causal_mask_length:, :-causal_mask_length] = True
                        # last causal_mask_length tokens can attend to all tokens before them (triangle shaped)
                        mask[-causal_mask_length:, -causal_mask_length:] = ~torch.triu(torch.ones(causal_mask_length, causal_mask_length, device=x.device).bool())
                        mask = mask.T
                    x, k, v = block(x, return_kv=True, mask=mask)
                k_list.append(k)
                v_list.append(v)
            # k_cache and v_cache have shape (n_layer, b, n_head, t, n_embd//n_head)
            k_cache = torch.stack(k_list, dim=0)
            v_cache = torch.stack(v_list, dim=0)

            x = self.transformer.ln_f(x)
            logits = self.lm_head(x[:, [-1]])
            # if sample range is not none, set all logits outside the range to -inf
            if sample_range is not None:
                logits[:, :, :sample_range[0]] = -float('inf')
                logits[:, :, sample_range[1]:] = -float('inf')
            # sample

            if i < num_unmasked_tokens:
                next_token = remaining_seq[i].view(1, 1)
            else:
                next_token = self.sample_logits(
                    logits,
                    temp=temperature[i],
                    top_k=top_k[i],
                    top_p=top_p[i],
                    blacklist=sampling_blacklist[i] if (sampling_blacklist is not None and len(sampling_blacklist) > i) else None
                )

            # if this is an index token (evey 17th) and patch_indexes is not None
            # replace the token with the corresponding patch index
            if i % n_tokens_per_patch == 0 and patch_indexes is not None:
                next_token = patch_indexes[:, i // n_tokens_per_patch].unsqueeze(-1)
            # append to the sequence and continue
            seq = torch.cat((seq, next_token), dim=1)

            all_logits.append(logits)

        if return_cache:
            return seq, k_cache, v_cache
        
        all_logits = torch.cat(all_logits, dim=1)

        return seq, all_logits

    @torch.no_grad()
    def rollout(
        self, 
        seq: torch.Tensor,
        pos: torch.Tensor,
        random_masked_indices,
        num_new_tokens: int,
        sampling_blacklist: List[List[int]] = None,
        temperature: Union[float, List[float]] = 1.0,
        top_k: Union[int, List[int]] = 1000,
        top_p: Union[float, List[float]] = 0.9,
        causal_mask_length: int = None,
        n_tokens_per_patch: int = None,
        num_unmasked_tokens=0,
        remaining_seq=None,
    ) -> Tuple[np.array, np.array, np.array]:
        """
        Rollout an arbitrary number of tokens from the given sequence

        Parameters:
            x (torch.LongTensor) of shape b, t: 
                The input image of shape b, t, where t <= 2*img_seq_len. The shorter that t is,
                the higher the masking ratio
            pos (LongTensor) of shape b, t:
                The positional indices of the sequence
            random_masked_indices (torch.Tensor) of shape b, t:
                The indices of the patches to reveal during generation
            temperature (int or List[int]):
                The temperature of the sampling (optionally per token)
            top_k (int or List[int]):
                The number of top k tokens to consider during sampling (optionally per token)
            top_p (float or List[float]):
                The cumulative probability threshold for nucleus (top-p) sampling (optionally per token)
            causal_mask_length (int):
                The length of the causal mask
        
        Returns:
            img1_pred (torch.LongTensor) of shape b, img_seq_len:
                The predicted frame 1
        """

        # get the device
        device = seq.device
        # get the dimensions
        b, t = seq.size()
        # get the number of new tokens to generate, it is the number of indexes in 
        # random_masked_indices multiplied by (1 + patch_size^2), since each index is
        # followed by a patch of size patch_size x patch_size
        if n_tokens_per_patch is None:
            t_gen = num_new_tokens // (1 + self.config.patch_size**2)
        else: # add the else condition to handle flow where the n_tokens_per_patch is computed differently
            t_gen = num_new_tokens // n_tokens_per_patch

        # trimp the random_masked_indices to the number of tokens to generate
        random_masked_indices = random_masked_indices[:, :t_gen]
        # figure out the total number of tokens in the sequence
        t_total = t + num_new_tokens

        # Pass the sequence through the model to predict the next frame
        # if we are not using classifier free guidance do a normal rollout
        pred_seq, logits = self.rollout_kv_cache(seq, num_new_tokens=num_new_tokens,
            temperature=temperature, patch_indexes=random_masked_indices, pos=pos, 
            top_k=top_k, top_p=top_p, causal_mask_length=causal_mask_length, 
            n_tokens_per_patch=n_tokens_per_patch, sampling_blacklist=sampling_blacklist,
            num_unmasked_tokens=num_unmasked_tokens, remaining_seq=remaining_seq)

        return pred_seq, logits 

    @torch.no_grad()
    def parallel_prediction(
            self, 
            seq: torch.Tensor, 
            pos: torch.Tensor,
            full_seq: torch.Tensor, 
            mask: torch.Tensor,
            n_tokens: int = 4148,
            causal_frame0: bool = False,
        ) -> torch.Tensor:
        """
        Forward pass of the model

        Parameters:
            seq (torch.Tensor) of size b, t: The input sequence
            pos (torch.Tensor) of size b, t: The positional indices of the sequence
            tgt (torch.Tensor) of size b, t_tgt: The target sequence
            mask (torch.Tensor) of size b, t, t: The mask of the sequence
        
        Returns:
            torch.Tensor: The logits of the model. Size b, t if tgt is None, else b, t_tgt
        """

        # grab device to perform operations on
        device = seq.device
        # grab dimensions
        b, t = seq.size()

        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # create a tensor of position indices, if not provided
        if pos is None:
            pos = torch.arange(t, device=device).unsqueeze(0).expand(b, -1)

        # forward the GPT model itself
        tok_emb = self.transformer.token_embedding(seq) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.positional_embedding(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # create attention mask
        with torch.no_grad():
            # Generate a mask template which will be used to mask out the attention
            num_patches_in_frame0 = self.config.rgb_patch_0_idx_range[1] - self.config.rgb_patch_0_idx_range[0]

            # The mask starts out with a "fully causal" mask, where each token can only attend to itself and tokens before it
            full_mask = torch.triu(torch.ones(t, t, 
                    device=x.device, requires_grad=False) * -float('inf'), diagonal=1).view(1, 1, t, t)
            # Next we set the "full attention cutoff" which is the number of leading tokens that can attend to each other
            # These tokens represent frame 0 and since we are not predicting them, they should be able to attend to each other
            if not causal_frame0:
                full_attention_cutoff = num_patches_in_frame0 * (1 + self.config.patch_size**2) # frame 0 has full attention
                full_mask[:, :, :full_attention_cutoff, :full_attention_cutoff] = 0
            # Finally we expand the provided custom sample specific mask
            unfolded_mask = torch.einsum('bn, bm -> bnm', mask, mask).view(b, 1, t, t)
            # We then combine the custom mask (which in practice specifies the attention patter of the lower right quadrant)
            # with the generic mask template to create custom masks
            mask = torch.masked_fill(full_mask, unfolded_mask.bool(), 0)
            
        for block in self.transformer.h:
            x = block(x, mask=mask)
        x = self.transformer.ln_f(x)
        
        # if tgt is not none, compute the logits and the loss for the target sequence
        logits = self.lm_head(x[:, -n_tokens:])
        
        # set all target tokens above 65535 to -1 so they are not included in the loss
        # we do this to ignore the prediction of the patch indexes since they are in random order
        # tgt[((65535+512) > tgt) & (tgt > 65535)] = -1

        sampled_tokens = logits.argmax(dim=-1) # self.sample_logits(logits, temp=1.0) # logits.argmax(dim=-1)
        full_pred_seq = torch.cat([seq[:, :-(n_tokens-1)], sampled_tokens], dim=1)

        full_pred_seq_patches = full_pred_seq.view(b, -1, (1 + self.config.patch_size**2))
        full_seq_pathces = full_seq.view(b, -1, (1 + self.config.patch_size**2))

        # set the position tokens of full_pred_seq_patches to the position tokens of full_seq_patches
        full_pred_seq_patches[:, :, 0] = full_seq_pathces[:, :, 0]

        # unfold the patches into a sequence
        full_pred_seq = full_pred_seq_patches.view(b, -1)

        # Unpack the predicted frame1 from flat seq and order the patches according to the random indices
        # To do this we first need to grab the number of tokens in frame 1 (both prediced and seeded)
        # so we can reconstruct the whole frame properly
        num_patches_in_frame1 = self.config.rgb_patch_1_idx_range[1] - self.config.rgb_patch_1_idx_range[0]
        num_tokens_in_frame1 = (1 + self.config.patch_size**2) * num_patches_in_frame1

        # add fake logits for revealed patches to the logits
        logits = torch.cat([torch.zeros(b, num_tokens_in_frame1-logits.shape[1], logits.shape[2], device=logits.device), logits], dim=1)

        # Grab the last num_tokens_in_frame1 tokens from the predicted sequence
        ordered_frame1_pred, ordered_frame1_logits = self.unpack_and_sort_img_seq(full_pred_seq[:, -num_tokens_in_frame1:], logits=logits)

        return ordered_frame1_pred, ordered_frame1_logits


    def unpack_img_seq(self, img_seq: torch.Tensor) -> torch.Tensor:
        """
        Unpacks a sequence of indices and image toknes into a 2d image tensor

        Parameters:
            img_seq (torch.Tensor) of size b, t:
                The input image sequence, where t is 17 * the number of image patches if patch_size is 4
        
        Returns:
            torch.Tensor: The unpacked image of size b, 16 * sqrt(t/17), 16 * sqrt(t/17) if patch_size is 4
        """

        # reshape the image sequence into a sequence of patches and 
        # trim the first element in the 2nd dim (the index)
        n_tokens_per_patch = self.config.patch_size**2 + 1 # compute the number of tokens per patch
        img_seq = img_seq.view(img_seq.size(0), -1, n_tokens_per_patch)[:, :, 1:]

        # unpatchify the image sequence
        img = unpatchify(img_seq)

        return img

    def unpack_and_sort_img_seq(self, img_seq: torch.Tensor, num_revealed_patches: int = 0, mark_revealed_patches: int = None, 
                                logits: torch.Tensor = None) -> torch.Tensor:
        """
        Unpacks a sequence of indices and image toknes into a 2d image tensor and sorts the patches
        according to the patch indices

        Parameters:
            img_seq (torch.Tensor) of size b, t:
                The input image sequence, where t is 17 * the number of image patches if patch_size is 4
            num_revealed_patches (int):
                The number of revealed patches to mark
            mark_revealed_patches (int):
                The value to mark the revealed patches with
            logits (torch.Tensor) of size b, t, n_tokens_per_patch, n_classes:
                The logits of the prediction
                
        Returns:
            torch.Tensor: The unpacked image of size b, 16 * sqrt(t/17), 16 * sqrt(t/17) if patch_size is 4
        """

        # reshape the image sequence into a sequence of patches and
        # trim the first element in the 2nd dim (the index)
        n_tokens_per_patch = self.config.patch_size**2 + 1 # compute the number of tokens per patch
        img_seq = img_seq.view(img_seq.size(0), -1, n_tokens_per_patch)
        img_idxs = img_seq[:, :, 0].long() - (65536 + 256) # shift the indices to the left by 65536 + 256
        reconstruct_indxs = torch.argsort(img_idxs, dim=1)
        rgb_seq = img_seq[:, :, 1:]

        # color the firs num_revealed_patches patches with the specified value if mark_revealed_patches is not None
        if mark_revealed_patches is not None:
            rgb_seq[:, :num_revealed_patches] = mark_revealed_patches

        # reorded the patches according to the patch indices
        rgb_seq = rgb_seq[torch.arange(img_seq.size(0)).unsqueeze(1).expand(*img_idxs.shape), reconstruct_indxs]

        # unpatchify the image sequence
        img = unpatchify(rgb_seq)

        if logits is not None:
            logits = logits.view(logits.size(0), -1, n_tokens_per_patch, logits.size(-1))
            logits = logits[:, :, 1:]
            logits = logits[torch.arange(img_seq.size(0)).unsqueeze(1).expand(*img_idxs.shape), reconstruct_indxs]
            logits = unpatchify_logits(logits)
            return img, logits
        
        return img

    def unpack_and_sort_flow_seq(
            self, 
            flow_seq_pred: torch.Tensor, 
            num_revealed_patches: int = 0,
            mark_revealed_patches: int = None,
            flow_size: int = 64
        ) -> torch.Tensor:
        
        """
        Unpacks a sequence of flow tokens into a sorted 2D flow tensor.

        Takes a sequence of flow tokens and patch indices and reconstructs the original 2D flow field,
        with optional marking of revealed patches.

        Parameters:
            flow_seq_pred: Predicted flow sequence tensor of shape (batch_size, num_tokens)
                Each patch contains patch_index followed by x,y flow values
            num_revealed_patches: Number of patches to mark as revealed (default: 0)
            mark_revealed_patches: Value to mark revealed patches with (default: None)
            flow_size: Size of flow field (default: 64)

        Returns:
            Tuple containing:
            - Flow tensor of shape (batch_size, 2, height, width) containing x,y flow components
            - Valid mask tensor of shape (batch_size, height, width) indicating which patches were predicted
        """
    
        batch_size = flow_seq_pred.shape[0]
        patch_size = 1 # self.config.patch_size
        num_patches = (flow_size // patch_size) ** 2
        tokens_per_patch = 3 # patch_size ** 2 * 2 + 1  # 1 patch index, 4 flow vectors per patch, 2 codes per vector (x,y)
        device = flow_seq_pred.device
        

        # Reshape sequence into patches and separate patch indices from flow values
        flow_seq_pred = flow_seq_pred.view(batch_size, -1, tokens_per_patch)
        patch_idx_pred = flow_seq_pred[:, :, 0] - self.config.flow_patch_idx_range[0]
        flow_code_pred = flow_seq_pred[:, :, 1:]

        # Mark revealed patches if specified
        if mark_revealed_patches is not None:
            flow_code_pred[:, :num_revealed_patches] = mark_revealed_patches

        # Reconstruct full flow codes of shape [B, num_patches, tokens_per_patch-1] by sorting patches
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(*patch_idx_pred.shape)
        flow_codes_full_shape = (batch_size, num_patches, tokens_per_patch - 1)

        if flow_code_pred.size(1) == num_patches:
            # If we have all patches, just sort them
            sort_indices = torch.argsort(patch_idx_pred, dim=1)
            flow_codes_full = flow_code_pred[batch_indices, sort_indices]
            valid_mask = torch.ones(flow_codes_full_shape).bool().to(device)
        else:
            # If some patches are missing, create a placeholder of shape [B, num_patches, tokens_per_patch-1] with zeros
            flow_codes_full = torch.zeros(flow_codes_full_shape)
            flow_codes_full = flow_codes_full.to(device).long()
            flow_codes_full[batch_indices, patch_idx_pred] = flow_code_pred
            valid_mask = torch.zeros(flow_codes_full_shape).bool().to(device)
            valid_mask[batch_indices, patch_idx_pred] = True

        # Separate and reshape x,y components
        flow_codes_full = flow_codes_full.view(batch_size, num_patches, patch_size ** 2, 2)
        flow_codes_full = flow_codes_full - self.config.flow_range[0]
        valid_mask = valid_mask.view(batch_size, num_patches, patch_size ** 2, 2)[..., 0]
        
        # Unpatchify x and y components separately
        flow_x = unpatchify(flow_codes_full[..., 0])
        flow_y = unpatchify(flow_codes_full[..., 1])
        valid_mask = unpatchify(valid_mask)

        # Stack into final flow tensor
        flow = torch.stack([flow_x, flow_y], dim=1)

        return flow, valid_mask
    
    def unpack_and_sort_campose_seq(self, campose_seq: torch.Tensor) -> torch.Tensor:
        """
        Unpacks a sequence of indices and campose tokens into a 2d tensor

        Parameters:
            campose_seq (torch.Tensor) of size b, t:
                The input campose sequence, where t is 6
        
        Returns:
            torch.Tensor: The unpacked campose of size b, 6
        """
        num_campose_pos_range = self.config.campose_pos_range[1] - self.config.campose_pos_range[0]
        # reshape the campose sequence into a sequence of campose tokens
        campose_seq_model_input = campose_seq.view(campose_seq.size(0), -1, num_campose_pos_range)[:, :, 1:]
        campose_seq_data = campose_seq_model_input - self.config.campose_range[0]

        return campose_seq_data


