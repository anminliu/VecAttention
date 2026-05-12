import argparse
import json
import os
import math
from copy import deepcopy

import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video
from termcolor import colored

from eval.DiTEvalKit.utils.dataloader import load_prompt_or_image
from eval.DiTEvalKit.models.wan.inference import replace_wan_attention
from eval.DiTEvalKit.models.diffusion_context import DiffusionStepContext
from eval.DiTEvalKit.utils.seed import seed_everything
from eval.DiTEvalKit.utils.timer import print_operator_log_data

from eval.DiTEvalKit.utils.logger import logger
from eval.DiTEvalKit.utils.misc import _load_threshold_pt
from spattn.src.models.config_args import add_fastprefill_args, get_fastprefill_config
from spattn.src.models.utils import SPARSITY_RECORD

print("[child] CUDA_VISIBLE_DEVICES =", os.getenv("CUDA_VISIBLE_DEVICES"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate video from text prompt using Wan-Diffuser")
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-14B-Diffusers", help="Model ID to use for generation")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for video generation")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative text prompt to avoid certain features")

    parser.add_argument("--prompt_source", type=str, default="prompt", choices=["prompt", "T2V_Wan_VBench", "T2V_Xingyang_VBench"], help="Source of the prompt")
    parser.add_argument("--prompt_idx", type=int, default=0, help="Index of the prompt")

    parser.add_argument("--height", type=int, default=720, help="Height of the generated video")
    parser.add_argument("--width", type=int, default=1280, help="Width of the generated video")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames in the generated video")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps in the generated video")
    parser.add_argument("--resolution", type=str, default="720p", choices=["480p", "720p"], help="Resolution of the generated video")
    parser.add_argument("--output_file", type=str, default="output.mp4", help="Output video file name")
    parser.add_argument("--logging_file", type=str, default=None, help="Path to the logging file.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for generation")
    parser.add_argument("--skip_existing", action="store_true", help="Skip generating existing output files")

    # parser.add_argument("--pattern", type=str, default="dense", choices=["SVG", "dense", "SAP"])
    parser.add_argument("--first_layers_fp", type=float, default=0.025, help="Only works for best config. Leave the 0, 1, 2, 40, 41 layers in FP")
    parser.add_argument("--first_times_fp", type=float, default=0.075, help="Only works for best config. Leave the first 10% timestep in FP")
    
    # SVG related
    parser.add_argument("--num_sampled_rows", type=int, default=64, help="The number of sampled rows")
    parser.add_argument("--sample_mse_max_row", type=int, default=10000, help="The maximum number of rows in attention mask. Prevent OOM.")
    parser.add_argument("--sparsity", type=float, default=0.25, help="The sparsity of the striped attention pattern. Accepts one or two float values.")

    # SAP related
    parser.add_argument("--num_q_centroids", "--qc", type=int, default=50, help="Number of query centroids for KMEANS_BLOCK.")
    parser.add_argument("--num_k_centroids", "--kc", type=int, default=200, help="Number of key centroids for KMEANS_BLOCK.")
    parser.add_argument("--top_p_kmeans", type=float, default=0.9, help="Top-p threshold for block selection in KMEANS_BLOCK.")
    parser.add_argument("--min_kc_ratio", type=float, default=0, help="At least this proportion of key blocks to keep per query block in KMEANS_BLOCK.")
    parser.add_argument("--kmeans_iter_init", type=int, default=0, help="Number of KMeans iterations for initialization in KMEANS_BLOCK.")
    parser.add_argument("--kmeans_iter_step", type=int, default=0, help="Number of KMeans iterations for other diffusion steps in KMEANS_BLOCK.")
    parser.add_argument("--zero_step_kmeans_init", action="store_true", help="Initialize the centroids for the first step in SAP, not after warmup.")


    # FastPrefill specific
    parser = add_fastprefill_args(parser)
    parser.add_argument("--threshold-path", type=str, default=None, help="Path to the base threshold pt file for FastPrefill.") 
    parser.add_argument("--neg-threshold-path", type=str, default=None, help="Path to the base threshold pt file for FastPrefill.")
    args = parser.parse_args()

    seed_everything(args.seed)

    # In some cases it will raise RuntimeError: cusolver error: CUSOLVER_STATUS_INTERNAL_ERROR
    torch.backends.cuda.preferred_linalg_library(backend="magma")
    
    if args.skip_existing:
        if os.path.exists(args.output_file):
            logger.info(f"Output file {args.output_file} already exists. Skipping generation.")
            exit(0)

    #########################################################
    # Load the model
    #########################################################
    # Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    model_id = args.model_id
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    flow_shift = 5.0  # 5.0 for 720P, 3.0 for 480P
    scheduler = UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
    pipe.scheduler = scheduler
        
    config = pipe.transformer.config
    
    #########################################################
    # Translate the percentage of warmup of layers and timesteps to the actual layers and timesteps
    #########################################################
    ref_scheduler = deepcopy(pipe.scheduler)
    ref_scheduler.set_timesteps(args.num_inference_steps)
    ref_timesteps = ref_scheduler.timesteps
    
    num_fp_timesteps = math.floor(args.first_times_fp * args.num_inference_steps)
    num_fp_layers = math.floor(args.first_layers_fp * config.num_layers)
    if num_fp_timesteps > 0:
        args.first_times_fp = ref_scheduler.timesteps[num_fp_timesteps - 1] - 1
    else:
        args.first_times_fp = 1001 # 1000 is the first timestep
    args.first_layers_fp = num_fp_layers
    
    logger.info(f"Warmup of Timesteps: {num_fp_timesteps} / {args.num_inference_steps} || {args.first_times_fp} / 1000 use FP")
    logger.info(f"Warmup of Layers: {num_fp_layers} / {config.num_layers} use FP")
    
    #########################################################
    # Load the prompt
    #########################################################
    args.prompt, _ = load_prompt_or_image(args.prompt_source, args.prompt_idx, args.prompt, None)

    if args.prompt is None:
        logger.info(colored("Using default prompt", "red"))
        args.prompt = "A cat walks on the grass, realistic"

    if args.negative_prompt is None:
        args.negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    print("=" * 20 + " Prompts " + "=" * 20)
    print(f"Prompt: {args.prompt}\n\n" + f"Negative Prompt: {args.negative_prompt}")

    #########################################################
    # Replace the attention & time logger
    #########################################################

    args.pattern = args.fastprefill_metric
    args.config = get_fastprefill_config(args)
    if args.pattern == "dense":
        args.config.metric = "full"

    if args.config.threshold is not None:
        args.config.pos_threshold = args.config.neg_threshold = args.config.threshold

    if args.threshold_path is not None and os.path.exists(args.threshold_path):
        args.config.pos_threshold = _load_threshold_pt(
            thr_path=args.threshold_path,
            num_layers=config.num_layers,
            num_heads=config.num_attention_heads,
        )
        if torch.cuda.is_available():
            args.config.pos_threshold = args.config.pos_threshold.cuda()
        logger.info(f"Loaded per-head pos_threshold matrix from {args.threshold_path}, pos_threshold shape: {args.config.pos_threshold.shape}")
    
    if args.neg_threshold_path is not None and os.path.exists(args.neg_threshold_path):
        args.config.neg_threshold = _load_threshold_pt(
            thr_path=args.neg_threshold_path,
            num_layers=config.num_layers,
            num_heads=config.num_attention_heads,
        )
        if torch.cuda.is_available():
            args.config.neg_threshold = args.config.neg_threshold.cuda()
        logger.info(f"Loaded per-head neg_threshold matrix from {args.neg_threshold_path}, neg_threshold shape: {args.config.neg_threshold.shape}")

    guidance_scale = 5.0
    if guidance_scale <= 1.0:
        if args.config.pos_threshold is not None:
            args.config.threshold = args.config.pos_threshold

    step_ctx = DiffusionStepContext(is_cfg=(guidance_scale > 1.0))

    assert args.config.pos_threshold is not None, "threshold or treshold_path are required."
    
    if args.pattern == "SVG":
        replace_wan_attention(
            pipe, 
            args.height, 
            args.width, 
            args.num_frames,
            first_layers_fp=args.first_layers_fp,
            first_times_fp=args.first_times_fp,
            pattern=args.pattern,
            # SVG specific
            num_sampled_rows=args.num_sampled_rows,
            sample_mse_max_row=args.sample_mse_max_row,
            sparsity=args.sparsity,
        )
    elif args.pattern == "SAP":
        replace_wan_attention(
            pipe,
            args.height,
            args.width,
            args.num_frames,
            first_layers_fp=args.first_layers_fp,
            first_times_fp=args.first_times_fp,
            pattern=args.pattern,
            # SAP specific
            num_q_centroids=args.num_q_centroids,
            num_k_centroids=args.num_k_centroids,
            top_p_kmeans=args.top_p_kmeans,
            min_kc_ratio=args.min_kc_ratio,
            logging_file=args.logging_file,
            kmeans_iter_init=args.kmeans_iter_init,
            kmeans_iter_step=args.kmeans_iter_step,
        )
    elif args.pattern in ["xattn", "vecattention", "dense"]:
        replace_wan_attention(
            pipe,
            args.height,
            args.width,
            args.num_frames,
            first_layers_fp=args.first_layers_fp,
            first_times_fp=args.first_times_fp,
            pattern=args.pattern,
            # fastprefill specific
            config=args.config,
            step_ctx=step_ctx,
        )
    else:
        assert args.pattern == "dense", f"Invalid pattern: {args.pattern}"

    # Print time logger
    for block in pipe.transformer.blocks:
        block.register_forward_hook(print_operator_log_data)

    pipe.to("cuda")

    #########################################################
    # Generate the video
    #########################################################
    output = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        guidance_scale=guidance_scale, 
        num_inference_steps=args.num_inference_steps
    ).frames[0]

    # Create parent directory for output file if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if hasattr(SPARSITY_RECORD, "attention_latency_list"):
        torch.cuda.synchronize()
        tot_latency = 0.0
        for start_evt, end_evt in SPARSITY_RECORD.attention_latency_list:
            latency = start_evt.elapsed_time(end_evt)  # milliseconds
            tot_latency += latency
        avg_latency = tot_latency / len(SPARSITY_RECORD.attention_latency_list)
        latency_file = os.path.join(output_dir, f'attention_latency_{args.config.chunk_size}.txt')
        with open(latency_file, 'w') as f:
            f.write(f"Average Attention Latency per layer: {avg_latency} ms\n")
            f.write(f"Total Attention Latency: {tot_latency} ms\n")
            for layer_idx, (start_evt, end_evt) in enumerate(SPARSITY_RECORD.attention_latency_list):
                latency = start_evt.elapsed_time(end_evt)
                f.write(f"Layer {layer_idx}: Attention Latency: {latency} ms\n")
        logger.info(f'Attention latency values saved to: {latency_file}')

    export_to_video(output, args.output_file, fps=16)
