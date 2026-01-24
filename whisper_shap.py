"""
SHAP utilities for Whisper-Flamingo

This module provides SHAP (SHapley Additive exPlanations) computation
for analyzing audio vs video contributions in Whisper-Flamingo models.
"""

import numpy as np
import torch
import shap
from typing import Tuple, Optional


def extract_features(
    model,
    mel: torch.Tensor,
    video: torch.Tensor,
    padding_mask: Optional[torch.Tensor] = None,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract audio and video features from the encoder.
    
    Args:
        model: Whisper-Flamingo model
        mel: Mel spectrogram (B, n_mels, T_mel)
        video: Video frames (B, C, T_frames, H, W)
        padding_mask: Padding mask for video (B, T_frames)
        device: Device for computation
        
    Returns:
        audio_features: (B, T_a, D_a)
        video_features: (B, T_v, D_v)
    """
    mel = mel.to(device)
    video = video.to(device)
    if padding_mask is not None:
        padding_mask = padding_mask.to(device)
    
    with torch.no_grad():
        audio_features, video_features = model.encoder(
            mel,
            x_v=video,
            training=False,
            test_a=False,
            test_v=False,
            padding_mask=padding_mask
        )
    
    audio_features = torch.zeros_like(audio_features)
    
    return audio_features, video_features


def generate_baseline_greedy(
    model,
    tokenizer,
    audio_features: torch.Tensor,
    video_features: torch.Tensor,
    max_length: int = 448,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Generate baseline tokens using greedy decoding.
    
    Follows Whisper's decoding pattern from decoding.py.
    
    Args:
        model: Whisper-Flamingo model
        tokenizer: Whisper tokenizer
        audio_features: Encoded audio features (B, T_a, D_a)
        video_features: Encoded video features (B, T_v, D_v)
        max_length: Maximum number of tokens to generate
        device: Device for computation
        
    Returns:
        Generated tokens WITHOUT SOT sequence (T_out,)
    """
    # Start with SOT sequence (like DecodingTask does)
    tokens = torch.tensor([tokenizer.sot_sequence], device=device)
    
    # Greedy generation - tokens ALWAYS includes SOT during generation
    for _ in range(max_length):
        with torch.no_grad():
            logits = model.decoder(
                tokens,
                audio_features,
                kv_cache=None,
                xv=video_features
            )
        
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=1)
        
        if next_token.item() == tokenizer.eot:
            break
    
    # Return WITHOUT SOT sequence (like Whisper does at decoding.py line 297-299)
    sot_len = len(tokenizer.sot_sequence)
    generated = tokens[0, sot_len:]
    
    return generated


def forward_shap_whisper_flamingo(
    model,
    tokenizer,
    mel: torch.Tensor,
    video: torch.Tensor,
    padding_mask: Optional[torch.Tensor] = None,
    nsamples: int = 2000,
    shap_alg: str = "kernel",
    device: str = 'cuda',
    verbose: bool = False
) -> Tuple[float, float, float, float, float, float]:
    """
    Compute SHAP values for Whisper-Flamingo.
    
    This follows the exact same structure as Llama-AVSR's forward_shap(),
    adapted for Whisper-Flamingo's architecture.
    
    Args:
        model: Whisper-Flamingo model
        tokenizer: Whisper tokenizer
        mel: Mel spectrogram (1, n_mels, T_mel)
        video: Video frames (1, C, T_frames, H, W)
        padding_mask: Padding mask for video (1, T_frames)
        nsamples: Number of SHAP samples
        shap_alg: SHAP algorithm ('kernel' or 'permutation')
        device: Device for computation
        verbose: Print debug information
        
    Returns:
        Tuple of 6 floats:
            (audio_pct_abs, video_pct_abs,
             audio_pct_pos, video_pct_pos,
             audio_pct_neg, video_pct_neg)
    """
    model.eval()
    
    # Verify single sample
    assert mel.shape[0] == 1, f"Expected batch size 1, got {mel.shape[0]}"
    assert video.shape[0] == 1, f"Expected batch size 1, got {video.shape[0]}"
    
    # 1. Extract features
    if verbose:
        print("Extracting features...")
    audio_features, video_features = extract_features(
        model, mel, video, padding_mask, device
    )
    
    # Store dimensions
    T_a = audio_features.shape[1]  # Audio timesteps
    T_v = video_features.shape[1]  # Video timesteps
    
    if verbose:
        print(f"  Audio features: {audio_features.shape}")
        print(f"  Video features: {video_features.shape}")
    
    # 2. Generate baseline tokens
    if verbose:
        print("Generating baseline tokens...")
    baseline_tokens_generated = generate_baseline_greedy(
        model, tokenizer, audio_features, video_features, device=device
    )
    
    if len(baseline_tokens_generated) == 0:
        raise ValueError("Baseline generation failed: no tokens generated")
    
    if verbose:
        baseline_text = tokenizer.decode(baseline_tokens_generated.tolist())
        print(f"  Baseline text: {baseline_text}")
        print(f"  Baseline tokens: {len(baseline_tokens_generated)}")
    
    # Create full sequence WITH SOT for teacher forcing
    sot_tensor = torch.tensor(tokenizer.sot_sequence, device=device)
    baseline_tokens_full = torch.cat([sot_tensor, baseline_tokens_generated])
    
    # 3. SHAP setup
    N_a = T_a
    N_v = T_v
    p = N_a + N_v
    
    if verbose:
        print(f"\nSHAP setup:")
        print(f"  Total features: {p} (audio: {N_a}, video: {N_v})")
    
    background = np.zeros((1, p), dtype=np.float32)
    x_explain = np.ones((1, p), dtype=np.float32)
    
    # 4. SHAP wrapper function
    def shap_wrapper(masks):
        return evaluate_coalitions_whisper_flamingo(
            model,
            masks,
            audio_features,
            video_features,
            baseline_tokens_generated,
            baseline_tokens_full,
            tokenizer,
            device
        )
    
    # 5. Compute SHAP
    if verbose:
        print(f"\nComputing SHAP with {nsamples} samples...")
    
    if shap_alg == "kernel":
        explainer = shap.SamplingExplainer(
            model=shap_wrapper,
            data=background
        )
        shap_values = explainer.shap_values(x_explain, nsamples=nsamples)
    
    elif shap_alg == "permutation":
        from shap.maskers import Independent
        masker = Independent(background, max_samples=100)
        
        explainer = shap.PermutationExplainer(
            model=shap_wrapper,
            masker=masker,
            algorithm='auto'
        )
        shap_obj = explainer(x_explain, max_evals=nsamples, silent=True)
        shap_values = shap_obj.values
    
    else:
        raise ValueError(f"Unknown SHAP algorithm: {shap_alg}")
    
    # 6. Process SHAP output
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    shap_values = np.array(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[0]
    
    vals = shap_values  # (p, T_out)
    
    if verbose:
        print(f"  SHAP values shape: {vals.shape}")
    
    # 7. Compute metrics (IDENTICAL to Llama-AVSR)
    # Absolute SHAP
    mm_raw_abs = np.sum(np.abs(vals), axis=1)
    mm_audio_abs = mm_raw_abs[:N_a].sum()
    mm_video_abs = mm_raw_abs[N_a:].sum()
    total_abs = mm_audio_abs + mm_video_abs
    
    audio_pct_abs = mm_audio_abs / total_abs if total_abs > 0 else 0.5
    video_pct_abs = mm_video_abs / total_abs if total_abs > 0 else 0.5
    
    # Positive SHAP
    mm_raw_pos = np.sum(np.maximum(vals, 0), axis=1)
    mm_audio_pos = mm_raw_pos[:N_a].sum()
    mm_video_pos = mm_raw_pos[N_a:].sum()
    total_pos = mm_audio_pos + mm_video_pos
    
    audio_pct_pos = mm_audio_pos / total_pos if total_pos > 0 else 0.5
    video_pct_pos = mm_video_pos / total_pos if total_pos > 0 else 0.5
    
    # Negative SHAP
    mm_raw_neg = np.sum(np.abs(np.minimum(vals, 0)), axis=1)
    mm_audio_neg = mm_raw_neg[:N_a].sum()
    mm_video_neg = mm_raw_neg[N_a:].sum()
    total_neg = mm_audio_neg + mm_video_neg
    
    audio_pct_neg = mm_audio_neg / total_neg if total_neg > 0 else 0.5
    video_pct_neg = mm_video_neg / total_neg if total_neg > 0 else 0.5
    
    if verbose:
        print(f"\nResults:")
        print(f"  Absolute - Audio: {audio_pct_abs*100:.2f}%, Video: {video_pct_abs*100:.2f}%")
        print(f"  Positive - Audio: {audio_pct_pos*100:.2f}%, Video: {video_pct_pos*100:.2f}%")
        print(f"  Negative - Audio: {audio_pct_neg*100:.2f}%, Video: {video_pct_neg*100:.2f}%")
    
    return (
        audio_pct_abs, video_pct_abs,
        audio_pct_pos, video_pct_pos,
        audio_pct_neg, video_pct_neg
    )


def evaluate_coalitions_whisper_flamingo(
    model,
    masks: np.ndarray,
    audio_features_full: torch.Tensor,
    video_features_full: torch.Tensor,
    baseline_tokens_generated: torch.Tensor,
    baseline_tokens_full: torch.Tensor,
    tokenizer,
    device: str
) -> np.ndarray:
    """
    SHAP wrapper: evaluate coalitions via teacher forcing.
    
    This follows the exact same structure as Llama-AVSR's shap_wrapper().
    
    Args:
        model: Whisper-Flamingo model
        masks: Binary mask array (n_coalitions, p) or (p,)
        audio_features_full: Full audio features (1, T_a, D_a)
        video_features_full: Full video features (1, T_v, D_v)
        baseline_tokens_generated: Generated tokens WITHOUT SOT (T_out,)
        baseline_tokens_full: Full tokens WITH SOT (sot_len + T_out,)
        tokenizer: Whisper tokenizer
        device: Device for computation
        
    Returns:
        results: Array of logits (n_coalitions, T_out)
    """
    if masks.ndim == 1:
        masks = masks.reshape(1, -1)
    
    n_coalitions = masks.shape[0]
    
    T_a = audio_features_full.shape[1]
    T_v = video_features_full.shape[1]
    
    results = []
    
    for i in range(n_coalitions):
        mask = masks[i]
        
        # Split mask: audio first, video second
        mask_audio = mask[:T_a]
        mask_video = mask[T_a:]
        
        # Clone and mask features
        audio_masked = audio_features_full.clone()
        video_masked = video_features_full.clone()
        
        # Zero out timesteps where mask=0
        for t in range(T_a):
            if mask_audio[t] == 0:
                audio_masked[:, t, :] = 0
        
        for t in range(T_v):
            if mask_video[t] == 0:
                video_masked[:, t, :] = 0
        
        # Teacher forcing with FULL sequence (including SOT)
        with torch.no_grad():
            logits = model.decoder(
                baseline_tokens_full.unsqueeze(0),  # Includes SOT
                audio_masked,
                kv_cache=None,
                xv=video_masked
            )
            
            # Extract logits for generated tokens
            # CRITICAL: logits[i] predicts token at position i+1
            # So logits[sot_len-1] predicts baseline_tokens_generated[0]
            sot_len = len(tokenizer.sot_sequence)
            logit_vec = []
            for t in range(len(baseline_tokens_generated)):
                token_id = baseline_tokens_generated[t].item()
                # Position in logits: sot_len - 1 + t (FIXED off-by-one bug)
                logit_vec.append(logits[0, sot_len - 1 + t, token_id].item())
            
            results.append(logit_vec)
    
    return np.array(results)