import argparse
import gc
import os
import time
import torch

from diffusers import OnnxStableDiffusionPipeline, StableDiffusionPipeline


def get_ort_pipeline(directory, provider):
    import onnxruntime

    if directory is not None:
        assert os.path.exists(directory)
        session_options = onnxruntime.SessionOptions()
        pipe = OnnxStableDiffusionPipeline.from_pretrained(
            directory,
            provider=provider,
            session_options=session_options,
        )
        return pipe

    # Original FP32 ONNX models
    pipe = OnnxStableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        revision="onnx",
        provider=provider,
        use_auth_token=True,
    )
    return pipe


def get_torch_pipeline(precision, unet_jit):
    if precision == "fp16":
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, revision=precision, use_auth_token=True
        ).to("cuda")
        pipe.unet_jit = unet_jit
    else:
        print("Skipping PyTorch FP32 for now")
        exit(1)
    return pipe


def profile_pipeline(pipe, batch_size):
    torch.backends.cudnn.benchmark = True

    height, width, num_inference_steps = 512, 512, 50
    prompts = ["a photo of an astronaut riding a horse on mars" for _ in range(batch_size)]
    with torch.inference_mode():
        # Warm up
        pipe(prompts, height, width, num_inference_steps=5)

        torch.cuda.synchronize()
        start = time.time()
        pipe(prompts, height, width, num_inference_steps)
        torch.cuda.synchronize()
        end = time.time()
        latency = end - start
        print(f"Batch size = {batch_size}, latency = {latency} s, throughput = {batch_size / latency} queries/s")

        # Garbage collect before measuring memory
        from onnxruntime.transformers.benchmark_helper import measure_memory

        gc.collect()
        torch.cuda.empty_cache()
        measure_memory(is_gpu=True, func=lambda: pipe(prompts, height, width, num_inference_steps))


def find_max_batch_size(pipe):
    # Warm up
    height, width = 512, 512
    num_inference_steps = 50
    min_batch_size, max_batch_size = 1, 1024
    while (min_batch_size <= max_batch_size):
        if isinstance(pipe, OnnxStableDiffusionPipeline):
            # Iterative search for maximum batch size
            batch_size = min_batch_size
        else:
            # Binary search for maximum batch size
            batch_size = min_batch_size + (max_batch_size - min_batch_size) // 2
        
        print(f"Attempting batch size = {batch_size}")
        try:
            prompts = ["a photo of an astronaut riding a horse on mars" for _ in range(batch_size)]
            start = time.time()
            pipe(prompts, height, width, num_inference_steps)
            end = time.time()
            latency = end - start
            print(f"Batch size = {batch_size}, latency = {latency} s, throughput = {batch_size / latency} queries/s")
            
            print(f"Batch size = {batch_size} is too low. Refining search space for min batch size.")
            if isinstance(pipe, OnnxStableDiffusionPipeline):
                min_batch_size += 1
            else:
                min_batch_size = batch_size+1
        except:
            print(f"Batch size = {batch_size} is too high. Refining search space for max batch size.")
            max_batch_size = batch_size-1

    print(f"Search is complete. Max batch size = {max_batch_size}.")


def choose_mode(pipe, mode, batch_size):
    if mode == "benchmark":
        assert batch_size > 0
        profile_pipeline(pipe, batch_size)
    else:
        find_max_batch_size(pipe)


def run_ort(directory, provider, mode, batch_size):
    load_start = time.time()
    pipe = get_ort_pipeline(directory, provider)
    load_end = time.time()
    print(f"Model loading took {load_end - load_start} seconds")
    choose_mode(pipe, mode, batch_size)


def run_torch(disable_conv_algo_search, precision, unet_jit, mode, batch_size):
    if not disable_conv_algo_search:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    load_start = time.time()
    pipe = get_torch_pipeline(precision, unet_jit)
    load_end = time.time()
    print(f"Model loading took {load_end - load_start} seconds")
    choose_mode(pipe, mode, batch_size)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-e",
        "--engine",
        required=False,
        type=str,
        default="onnxruntime",
        choices=["onnxruntime", "torch"],
        help="Engines to benchmark",
    )

    parser.add_argument(
        "-p",
        "--pipeline",
        required=False,
        type=str,
        default=None,
        help="Directory of saved onnx pipeline for CompVis/stable-diffusion-v1-4.",
    )

    parser.add_argument(
        "-d",
        "--disable_conv_algo_search",
        required=False,
        action="store_true",
        help="Disable cuDNN conv algo search.",
    )
    parser.set_defaults(disable_conv_algo_search=False)

    parser.add_argument(
        "-f",
        "--floating_point_precision",
        required=False,
        type=str,
        default="fp16",
        help="Floating point precision of model",
    )

    parser.add_argument(
        "-u",
        "--unet_jit",
        required=False,
        action="store_true",
        help="Use unet torchscript",
    )
    parser.set_defaults(unet_jit=False)

    parser.add_argument(
        "-b",
        "--batch_size",
        required=False,
        type=int,
        default=0,
    )

    parser.add_argument(
        "-m",
        "--mode",
        required=True,
        type=str,
        help="Mode to evaluate pipeline on",
        choices=["benchmark", "search"]
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    if args.engine == "onnxruntime":
        provider = (
            ["CUDAExecutionProvider"]
            if args.disable_conv_algo_search
            else [
                (
                    "CUDAExecutionProvider",
                    {"cudnn_conv_use_max_workspace": "1", "cudnn_conv_algo_search": "EXHAUSTIVE"},
                ),
                #"CPUExecutionProvider",
            ]
        )
        run_ort(args.pipeline, provider, args.mode, args.batch_size)
    else:
        run_torch(args.disable_conv_algo_search, args.floating_point_precision, args.unet_jit, args.mode, args.batch_size)


if __name__ == "__main__":
    main()
