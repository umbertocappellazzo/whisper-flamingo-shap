"""
Whisper-Flamingo SHAP Inference Script

Modified version of whisper_decode_video.py for computing SHAP values
instead of WER/BLEU metrics.

This script:
1. Loads a Whisper-Flamingo model
2. Iterates through test samples
3. Computes SHAP values for each sample
4. Logs results to WandB
5. Saves aggregate statistics
"""

import os
import json
import argparse
import numpy as np
import torch
import whisper
from tqdm import tqdm
from utils import (
    load_data,
    WhisperVideoCollatorWithPadding,
)
from utils_batch_samplers import LengthBatchSampler
from whisper_ft_muavic_video import MuavicVideoDataset
from whisper_shap import forward_shap_whisper_flamingo


import wandb


parser = argparse.ArgumentParser(description='Whisper-Flamingo SHAP Analysis')

# Model arguments
parser.add_argument('--lang', default='en', type=str, help='Language')
parser.add_argument('--model-type', default='medium', help='Whisper model size')
parser.add_argument('--checkpoint-path', required=True, help='Path to Whisper-Flamingo checkpoint')
parser.add_argument('--whisper-path', default="models/", help='Path to Whisper models')
parser.add_argument('--av-hubert-path', default="av_hubert/avhubert/", help='Path to AV-HuBERT code')
parser.add_argument('--av-hubert-ckpt', default="models/large_noise_pt_noise_ft_433h_only_weights.pt",
                    help='Path to AV-HuBERT checkpoint')

# Architecture arguments
parser.add_argument('--use_av_hubert_encoder', default=1, type=int, help='Use AV-HuBERT encoder')
parser.add_argument('--av_fusion', default="separate", help='Audio-video fusion method')
parser.add_argument('--add_gated_x_attn', default=1, type=int, help='Add gated cross-attention')
parser.add_argument('--fp16', default=1, type=int, help='Use FP16')

# SHAP arguments
parser.add_argument('--num-samples-shap', default=2000, type=int, help='Number of SHAP samples')
parser.add_argument('--shap-alg', default='kernel', choices=['kernel', 'permutation'],
                    help='SHAP algorithm')
parser.add_argument('--verbose', action='store_true', 
                    help='Print detailed SHAP info')
parser.add_argument('--debug', action='store_true', 
                    help='Enable extensive debugging output (more detailed than --verbose)')

# Data arguments
parser.add_argument('--noise-snr', default=1000, type=float, help='>100 is off (clean audio)')
parser.add_argument('--noise-fn', default=None, help='Noise file for testing')
parser.add_argument('--modalities', default="avsr", help='asr/avsr/vsr')

# Output arguments
#parser.add_argument('--output-path', default="shap_results/", help='Path to save results')

# WandB arguments
parser.add_argument('--wandb-project', default=None, type=str, 
                    help='WandB project name (if None, WandB disabled)')
parser.add_argument('--exp-name', default=None, type=str,
                    help='WandB run name (auto-generated if None)')
parser.add_argument('--output-path', default=None, type=str)

args = parser.parse_args()

SAMPLE_RATE = 16000
SEED = 3407
torch.manual_seed(SEED)

# Process language
use_lrs2 = True if args.lang == 'lrs2' else False
if args.lang == 'lrs2':
    args.lang = 'en'

# Load data
print("Loading data...")
audio_transcript_pair_list = load_data(
    480000, 350, [args.lang],
    muavic_root='',
    include_audio_lens=True,
    task='transcribe',
    lrs2=use_lrs2
)

test_dataset = audio_transcript_pair_list['test']
test_dataset = [[i[0], i[1].replace('/data/sls/scratch/roudi/datasets/muavic/', ''),
                 i[2], i[3]] for i in test_dataset]

print(f"Test dataset size: {len(test_dataset)}")

print("Test # 0: ", test_dataset[0])
print("Test # 15: ", test_dataset[15])
print("Test # 330: ", test_dataset[300])

# Create tokenizer
multilingual = True if 'large' in args.model_type or 'en' not in args.model_type else False
print(f"Multilingual tokenizer: {multilingual}")
tokenizer = whisper.tokenizer.get_tokenizer(multilingual=multilingual, task='transcribe')
special_token_set = set(tokenizer.special_tokens.values())

# Create dataset
dataset = MuavicVideoDataset(
    test_dataset,
    tokenizer,
    SAMPLE_RATE,
    args.model_type,
    max_length=None,
    spec_augment="",
    noise_prob=1 if args.noise_snr != 1000 else 0,
    noise_fn=args.noise_fn,
    train=False,
    noise_snr=args.noise_snr,
)

# Create dataloader - SINGLE SAMPLE batches for SHAP
length_sorter = LengthBatchSampler(
    batch_bins=1,  # Single sample per batch
    shapes=[i[3] for i in test_dataset],
    sort_in_batch='descending',
    sort_batch='descending',
    drop_last=False
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=4,
    collate_fn=WhisperVideoCollatorWithPadding(),
    batch_sampler=length_sorter
)

print("Loading Whisper-Flamingo model...")
whisper_model = whisper.load_model(
    args.model_type,
    download_root=args.whisper_path,
    video=True if args.av_fusion != "None" else 0,
    video_model_path=args.av_hubert_ckpt,
    av_hubert_path=args.av_hubert_path,
    av_hubert_encoder=args.use_av_hubert_encoder,
    av_fusion=args.av_fusion,
    add_gated_x_attn=args.add_gated_x_attn
)

if args.checkpoint_path is not None:
    print(f"Loading checkpoint: {args.checkpoint_path}")
    state_dict = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    state_dict_updated = {k[6:]: v for k, v in state_dict.items()}  # remove 'model.'
    try:
        whisper_model.load_state_dict(state_dict_updated)
    except BaseException as e:
        print(str(e))
        print("Loading weights with strict=False")
        whisper_model.load_state_dict(state_dict_updated, strict=False)

# Convert to fp16 if needed
if args.fp16 and args.use_av_hubert_encoder == 1:
    whisper_model.encoder.video_projection_scalar.half()
    whisper_model.encoder.video_model.half()
    model_to_num_layers = {'small': 12, 'medium': 24, 'medium.en': 24, 'large-v2': 32}
    if args.av_fusion == 'separate':
        for i in range(model_to_num_layers[args.model_type]):
            try:
                whisper_model.decoder.blocks[i].attn_gate.data = \
                    whisper_model.decoder.blocks[i].attn_gate.half()
                whisper_model.decoder.blocks[i].ff_gate.data = \
                    whisper_model.decoder.blocks[i].ff_gate.half()
            except:
                continue

# Move to device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
whisper_model = whisper_model.to(device)
whisper_model.eval()

print(f"Model loaded on {device}")

wandb.init(
    project=args.wandb_project,
    name = args.exp_name
)

# Create output directory
# os.makedirs(args.output_path, exist_ok=True)

# Storage for results
results = {
    'audio_abs': [],
    'video_abs': [],
    'audio_pos': [],
    'video_pos': [],
    'audio_neg': [],
    'video_neg': [],
    'num_audio_tokens': [],
    'shapley_values': [],
    'baseline_texts': [],
}

# Run SHAP analysis
print("\n" + "="*80)
print("Starting SHAP Analysis")
print("="*80)

for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing SHAP")):
    if args.fp16:
        input_ids = batch["input_ids"].half().cuda()
        video = batch["video"].half().cuda()
        padding_mask = batch["padding_mask"].cuda()  # ADD THIS
    else:
        input_ids = batch["input_ids"].to(device)
        video = batch["video"].to(device)
        padding_mask = batch["padding_mask"].to(device)  # ADD THIS
    
    labels = batch["labels"]
    
    try:
        # Compute SHAP values
        (audio_abs, video_abs,
         audio_pos, video_pos,
         audio_neg, video_neg,
         num_audio_tokens, shapley_values) = forward_shap_whisper_flamingo(
            model=whisper_model,
            tokenizer=tokenizer,
            mel=input_ids,
            video=video,
            padding_mask=padding_mask,
            nsamples=args.num_samples_shap,
            shap_alg=args.shap_alg,
            device=device,
            verbose=args.verbose,
            debug=args.debug
        )
        
        # Store results
        results['audio_abs'].append(audio_abs)
        results['video_abs'].append(video_abs)
        results['audio_pos'].append(audio_pos)
        results['video_pos'].append(video_pos)
        results['audio_neg'].append(audio_neg)
        results['video_neg'].append(video_neg)
        results['num_audio_tokens'].append(num_audio_tokens)
        results['shapley_values'].append(shapley_values)
        
        # Decode reference text
        labels_clean = labels.clone()
        labels_clean[labels_clean == -100] = tokenizer.eot
        ref_text = tokenizer.decode([t for t in labels_clean[0] 
                                     if t.item() not in special_token_set])
        
        wandb.log({
            'sample_idx': batch_idx,
            'sample_audio_abs': audio_abs,
            'sample_video_abs': video_abs,
            'sample_audio_pos': audio_pos,
            'sample_video_pos': video_pos,
            'sample_audio_neg': audio_neg,
            'sample_video_neg': video_neg,
            'sample_num_audio_tokens': num_audio_tokens
        })
        
        print(f"\nSample {batch_idx + 1}/{len(dataloader)}:")
        print(f"  Absolute - Audio: {audio_abs*100:.2f}%, Video: {video_abs*100:.2f}%")
        print(f"  Reference: {ref_text[:100]}...")
        
    
    except Exception as e:
        print(f"\nError processing sample {batch_idx}: {e}")
        continue

# Compute aggregate statistics
print("\n" + "="*80)
print("SHAP Analysis Complete")
print("="*80)

mean_audio_abs = np.mean(results['audio_abs'])
mean_video_abs = np.mean(results['video_abs'])
mean_audio_pos = np.mean(results['audio_pos'])
mean_video_pos = np.mean(results['video_pos'])
mean_audio_neg = np.mean(results['audio_neg'])
mean_video_neg = np.mean(results['video_neg'])
mean_num_audio_tokens = np.mean(results['num_audio_tokens'])

std_audio_abs = np.std(results['audio_abs'])
std_video_abs = np.std(results['video_abs'])

print(f"\nAggregate Results (n={len(results['audio_abs'])}):")
print(f"\nAbsolute SHAP:")
print(f"  Audio: {mean_audio_abs*100:.2f}% ± {std_audio_abs*100:.2f}%")
print(f"  Video: {mean_video_abs*100:.2f}% ± {std_video_abs*100:.2f}%")
print(f"\nPositive SHAP:")
print(f"  Audio: {mean_audio_pos*100:.2f}%")
print(f"  Video: {mean_video_pos*100:.2f}%")
print(f"\nNegative SHAP:")
print(f"  Audio: {mean_audio_neg*100:.2f}%")
print(f"  Video: {mean_video_neg*100:.2f}%")

wandb.log({
    'audio-ABS-SHAP': mean_audio_abs,
    'video-ABS-SHAP': mean_video_abs,
    'audio-POS-SHAP': mean_audio_pos,
    'video-POS-SHAP': mean_video_pos,
    'audio-NEG-SHAP': mean_audio_neg,
    'video-NEG-SHAP': mean_video_neg,
    'audio-ABS-STD': std_audio_abs,
    'video-ABS-STD': std_video_abs,
    'num-audio-tokens': mean_num_audio_tokens
})



output_file = os.path.join(
    args.output_path,
    args.exp_name
    
)

print("Output dir: ", output_file)

np.savez_compressed(
        output_file,
        # Aggregated metrics
        audio_abs=np.array(results['audio_abs']),
        video_abs=np.array(results['video_abs']),
        audio_pos=np.array(results['audio_pos']),
        video_pos=np.array(results['video_pos']),
        audio_neg=np.array(results['audio_neg']),
        video_neg=np.array(results['video_neg']),
        num_audio_tokens=np.array(results['num_audio_tokens']),
        
        # Raw SHAP values (ragged array - stored as object array)
        shap_values=np.array(results['shapley_values'], dtype=object),
    )