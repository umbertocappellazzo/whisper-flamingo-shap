"""
SHAP utilities for Whisper-Flamingo with comprehensive debugging
"""

import numpy as np
import torch
import shap
from typing import Tuple, Optional
import warnings


def extract_features(
    model,
    mel: torch.Tensor,
    video: torch.Tensor,
    padding_mask: Optional[torch.Tensor] = None,
    device: str = 'cuda',
    debug: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract audio and video features from the encoder."""
    if debug:
        print(f"\n[DEBUG extract_features]")
        print(f"  Input mel shape: {mel.shape}, dtype: {mel.dtype}")
        print(f"  Input video shape: {video.shape}, dtype: {video.dtype}")
        if padding_mask is not None:
            print(f"  Padding mask shape: {padding_mask.shape}")
    
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
    
    # Validate outputs
    if torch.isnan(audio_features).any():
        warnings.warn("NaN detected in audio_features!")
    if torch.isnan(video_features).any():
        warnings.warn("NaN detected in video_features!")
    
    if debug:
        print(f"  Output audio_features: {audio_features.shape}, range: [{audio_features.min():.4f}, {audio_features.max():.4f}]")
        print(f"  Output video_features: {video_features.shape}, range: [{video_features.min():.4f}, {video_features.max():.4f}]")
        print(f"  Audio features zero ratio: {(audio_features == 0).float().mean():.4f}")
        print(f"  Video features zero ratio: {(video_features == 0).float().mean():.4f}")
    
    return audio_features, video_features


def generate_baseline_greedy(
    model,
    tokenizer,
    audio_features: torch.Tensor,
    video_features: torch.Tensor,
    max_length: int = 448,
    device: str = 'cuda',
    debug: bool = False
) -> torch.Tensor:
    """Generate baseline tokens using greedy decoding."""
    if debug:
        print(f"\n[DEBUG generate_baseline_greedy]")
        print(f"  Max length: {max_length}")
        print(f"  SOT sequence: {tokenizer.sot_sequence}")
    
    # Start with SOT sequence
    tokens = torch.tensor([tokenizer.sot_sequence], device=device)
    
    if debug:
        print(f"  Initial tokens: {tokens.tolist()}")
    
    # Greedy generation
    generated_count = 0
    for step in range(max_length):
        with torch.no_grad():
            logits = model.decoder(
                tokens,
                audio_features,
                kv_cache=None,
                xv=video_features
            )
        
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=1)
        generated_count += 1
        
        if debug and step < 5:
            print(f"  Step {step}: predicted token {next_token.item()}")
        
        if next_token.item() == tokenizer.eot:
            if debug:
                print(f"  EOT reached at step {step}")
            break
    
    # Return WITHOUT SOT sequence
    sot_len = len(tokenizer.sot_sequence)
    generated = tokens[0, sot_len:]
    
    if debug:
        print(f"  Total tokens generated: {len(generated)}")
        print(f"  First 10 tokens: {generated[:10].tolist()}")
        print(f"  Last 5 tokens: {generated[-5:].tolist()}")
    
    return generated


def evaluate_coalitions_whisper_flamingo(
    model,
    masks: np.ndarray,
    audio_features_full: torch.Tensor,
    video_features_full: torch.Tensor,
    baseline_tokens_generated: torch.Tensor,
    baseline_tokens_full: torch.Tensor,
    tokenizer,
    device: str,
    debug: bool = False,
    coalition_idx: int = 0
) -> np.ndarray:
    """
    SHAP wrapper: evaluate coalitions via teacher forcing.
    
    Returns a VECTOR of logits per coalition (matching Llama-AVSR).
    """
    if masks.ndim == 1:
        masks = masks.reshape(1, -1)
    
    n_coalitions = masks.shape[0]
    T_a = audio_features_full.shape[1]
    T_v = video_features_full.shape[1]
    
    if debug and coalition_idx == 0:
        print(f"\n[DEBUG evaluate_coalitions]")
        print(f"  Number of coalitions: {n_coalitions}")
        print(f"  Audio timesteps: {T_a}, Video timesteps: {T_v}")
        print(f"  Baseline tokens (generated): {len(baseline_tokens_generated)}")
        print(f"  Baseline tokens (full with SOT): {len(baseline_tokens_full)}")
        print(f"  SOT length: {len(tokenizer.sot_sequence)}")
    
    results = []
    
    for i in range(n_coalitions):
        mask = masks[i]
        
        # Debug first coalition
        if debug and i == 0:
            print(f"\n  Coalition 0 analysis:")
            print(f"    Mask shape: {mask.shape}")
            print(f"    Audio features kept: {mask[:T_a].sum()}/{T_a} ({mask[:T_a].mean()*100:.1f}%)")
            print(f"    Video features kept: {mask[T_a:].sum()}/{T_v} ({mask[T_a:].mean()*100:.1f}%)")
        
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
        
        if debug and i == 0:
            print(f"    Audio masked zero ratio: {(audio_masked == 0).float().mean():.4f}")
            print(f"    Video masked zero ratio: {(video_masked == 0).float().mean():.4f}")
        
        # Teacher forcing with FULL sequence (including SOT)
        with torch.no_grad():
            logits = model.decoder(
                baseline_tokens_full.unsqueeze(0),
                audio_masked,
                kv_cache=None,
                xv=video_masked
            )
            
            if debug and i == 0:
                print(f"    Decoder output logits shape: {logits.shape}")
                print(f"    Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
            
            # Extract logits for generated tokens
            # logits[i] predicts token at position i+1
            # So logits[sot_len-1] predicts baseline_tokens_generated[0]
            sot_len = len(tokenizer.sot_sequence)
            
            # Build positions tensor
            T = len(baseline_tokens_generated)
            positions = sot_len - 1 + torch.arange(T, device=device)
            
            # Validate positions
            if debug and i == 0:
                print(f"    Extracting logits at positions: {positions[:5].tolist()} ... {positions[-3:].tolist()}")
                print(f"    For tokens: {baseline_tokens_generated[:5].tolist()} ... {baseline_tokens_generated[-3:].tolist()}")
                max_pos = positions.max().item()
                if max_pos >= logits.shape[1]:
                    raise ValueError(
                        f"Position index {max_pos} out of bounds for logits shape {logits.shape}"
                    )
            
            # Extract logits for baseline tokens (matching Llama-AVSR)
            logit_vec = logits[0, positions, baseline_tokens_generated]
            
            if debug and i == 0:
                print(f"    Extracted logit_vec shape: {logit_vec.shape}")
                print(f"    Logit values range: [{logit_vec.min():.4f}, {logit_vec.max():.4f}]")
                print(f"    First 5 logits: {logit_vec[:5].tolist()}")
                print(f"    Last 3 logits: {logit_vec[-3:].tolist()}")
            
            logit_vec_np = logit_vec.detach().cpu().numpy()
            
            # Check for invalid values
            if np.isnan(logit_vec_np).any():
                warnings.warn(f"NaN detected in coalition {i} logits!")
            if np.isinf(logit_vec_np).any():
                warnings.warn(f"Inf detected in coalition {i} logits!")
            
            results.append(logit_vec_np)
    
    # Return (n_coalitions, T_out) - VECTORS not scalars!
    result_array = np.array(results)
    
    if debug:
        print(f"\n  Final results array shape: {result_array.shape}")
        print(f"  Results range: [{result_array.min():.4f}, {result_array.max():.4f}]")
    
    return result_array


def forward_shap_whisper_flamingo(
    model,
    tokenizer,
    mel: torch.Tensor,
    video: torch.Tensor,
    padding_mask: Optional[torch.Tensor] = None,
    nsamples: int = 2000,
    shap_alg: str = "kernel",
    device: str = 'cuda',
    verbose: bool = False,
    debug: bool = False
) -> Tuple[float, float, float, float, float, float]:
    """
    Compute SHAP values for Whisper-Flamingo.
    
    Args:
        debug: Enable extensive debugging output (more than verbose)
    """
    model.eval()
    
    # Verify single sample
    assert mel.shape[0] == 1, f"Expected batch size 1, got {mel.shape[0]}"
    assert video.shape[0] == 1, f"Expected batch size 1, got {video.shape[0]}"
    
    if debug:
        print("\n" + "="*80)
        print("SHAP COMPUTATION DEBUG MODE")
        print("="*80)
    
    # 1. Extract features
    if verbose or debug:
        print("\n[1] Extracting features...")
    audio_features, video_features = extract_features(
        model, mel, video, padding_mask, device, debug=debug
    )
    
    T_a = audio_features.shape[1]
    T_v = video_features.shape[1]
    
    if verbose or debug:
        print(f"  Audio features: {audio_features.shape}")
        print(f"  Video features: {video_features.shape}")
    
    # 2. Generate baseline tokens
    if verbose or debug:
        print("\n[2] Generating baseline tokens...")
    baseline_tokens_generated = generate_baseline_greedy(
        model, tokenizer, audio_features, video_features, 
        device=device, debug=debug
    )
    
    if len(baseline_tokens_generated) == 0:
        raise ValueError("Baseline generation failed: no tokens generated")
    
    if verbose or debug:
        # Filter special tokens for display
        special_token_set = set(tokenizer.special_tokens.values())
        filtered_tokens = [t for t in baseline_tokens_generated.tolist() 
                          if t not in special_token_set]
        baseline_text = tokenizer.decode(filtered_tokens)
        print(f"  Baseline text: {baseline_text}")
        print(f"  Baseline tokens: {len(baseline_tokens_generated)}")
    
    # Create full sequence WITH SOT for teacher forcing
    sot_tensor = torch.tensor(tokenizer.sot_sequence, device=device)
    baseline_tokens_full = torch.cat([sot_tensor, baseline_tokens_generated])
    
    if debug:
        print(f"\n  Baseline construction:")
        print(f"    SOT sequence length: {len(sot_tensor)}")
        print(f"    Generated tokens length: {len(baseline_tokens_generated)}")
        print(f"    Full baseline length: {len(baseline_tokens_full)}")
        print(f"    Full baseline: {baseline_tokens_full[:10].tolist()} ... {baseline_tokens_full[-5:].tolist()}")
    
    # 3. SHAP setup
    N_a = T_a
    N_v = T_v
    p = N_a + N_v
    
    if verbose or debug:
        print(f"\n[3] SHAP setup:")
        print(f"  Total features: {p} (audio: {N_a}, video: {N_v})")
    
    background = np.zeros((1, p), dtype=np.float32)
    x_explain = np.ones((1, p), dtype=np.float32)
    
    if debug:
        print(f"  Background (all removed): {background.shape}")
        print(f"  Explain (all present): {x_explain.shape}")
    
    # 4. SHAP wrapper function
    coalition_counter = [0]  # Mutable counter for debugging
    
    def shap_model(masks):
        result = evaluate_coalitions_whisper_flamingo(
            model,
            masks,
            audio_features,
            video_features,
            baseline_tokens_generated,
            baseline_tokens_full,
            tokenizer,
            device,
            debug=debug,
            coalition_idx=coalition_counter[0]
        )
        coalition_counter[0] += masks.shape[0] if masks.ndim > 1 else 1
        
        # # DEBUG: Print first call to see what SHAP receives
        # if not hasattr(shap_model, 'called'):
        #     print(f"\n[SHAP WRAPPER DEBUG]")
        #     print(f"  Wrapper receives masks shape: {masks.shape}")
        #     print(f"  Wrapper returns result shape: {result.shape}")
        #     print(f"  Result dtype: {result.dtype}")
        #     shap_model.called = True
        
        return result
    
    
    
    # 5. Compute SHAP
    if verbose or debug:
        print(f"\n[4] Computing SHAP with {nsamples} samples using {shap_alg}...")
    
    if shap_alg == "kernel":
        explainer = shap.SamplingExplainer(
            model=shap_model,
            data=background
        )
        shap_values_raw = explainer.shap_values(x_explain, nsamples=nsamples)
        
        # DEBUG: See what SHAP returns
        print(f"\n[SHAP OUTPUT DEBUG]")
        print(f"  Type: {type(shap_values_raw)}")
        if isinstance(shap_values_raw, list):
            print(f"  List length: {len(shap_values_raw)}")
            print(f"  First element type: {type(shap_values_raw[0])}")
            print(f"  First element shape: {np.array(shap_values_raw[0]).shape}")
            if len(shap_values_raw) > 1:
                print(f"  Second element shape: {np.array(shap_values_raw[1]).shape}")
        else:
            print(f"  Array shape: {np.array(shap_values_raw).shape}")
        
        shap_values = shap_values_raw
    
    elif shap_alg == "permutation":
        from shap.maskers import Independent
        masker = Independent(background, max_samples=100)
        
        explainer = shap.PermutationExplainer(
            model=shap_model,
            masker=masker,
            algorithm='auto'
        )
        shap_obj = explainer(x_explain, max_evals=nsamples, silent=True)
        shap_values = shap_obj.values
    
    else:
        raise ValueError(f"Unknown SHAP algorithm: {shap_alg}")
    
    #if debug:
    print(f"\n  Total coalitions evaluated: {coalition_counter[0]}")
    
    # 6. Process SHAP output (MATCHING Llama-AVSR EXACTLY)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    shap_values = np.array(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[0]
    
    # CRITICAL: SHAP returns (p, T) - one row per feature, one col per token
    vals = shap_values
    
    if verbose or debug:
        print(f"\n[5] SHAP values processing:")
        print(f"  SHAP values shape: {vals.shape}")
        print(f"  Expected shape: ({p}, {len(baseline_tokens_generated)})")
    
    # Validate shape
    expected_shape = (p, len(baseline_tokens_generated))
    if vals.shape != expected_shape:
        warnings.warn(
            f"SHAP values shape {vals.shape} doesn't match expected {expected_shape}!"
        )
    
    if debug:
        print(f"  SHAP values range: [{vals.min():.4f}, {vals.max():.4f}]")
        print(f"  SHAP values mean: {vals.mean():.4f}, std: {vals.std():.4f}")
        print(f"  Audio SHAP values (first feature): {vals[0, :5]}")
        print(f"  Video SHAP values (first feature): {vals[N_a, :5]}")
        
        # Check for anomalies
        if np.isnan(vals).any():
            warnings.warn("NaN values detected in SHAP values!")
        if np.isinf(vals).any():
            warnings.warn("Inf values detected in SHAP values!")
    
    # 7. Compute metrics (IDENTICAL to Llama-AVSR)
    # Absolute SHAP - sum over tokens (axis=1)
    mm_raw_abs = np.sum(np.abs(vals), axis=1)  # (p,)
    mm_audio_abs = mm_raw_abs[:N_a].sum()
    mm_video_abs = mm_raw_abs[N_a:].sum()
    total_abs = mm_audio_abs + mm_video_abs
    
    if debug:
        print(f"\n[6] Computing metrics:")
        print(f"  Audio absolute contribution: {mm_audio_abs:.4f}")
        print(f"  Video absolute contribution: {mm_video_abs:.4f}")
        print(f"  Total absolute: {total_abs:.4f}")
    
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
    
    if verbose or debug:
        print(f"\n[7] Final Results:")
        print(f"  Absolute - Audio: {audio_pct_abs*100:.2f}%, Video: {video_pct_abs*100:.2f}%")
        print(f"  Positive - Audio: {audio_pct_pos*100:.2f}%, Video: {video_pct_pos*100:.2f}%")
        print(f"  Negative - Audio: {audio_pct_neg*100:.2f}%, Video: {video_pct_neg*100:.2f}%")
    
    if debug:
        print("="*80)
        print("SHAP COMPUTATION COMPLETE")
        print("="*80 + "\n")
    
    return (
        audio_pct_abs, video_pct_abs,
        audio_pct_pos, video_pct_pos,
        audio_pct_neg, video_pct_neg
    )