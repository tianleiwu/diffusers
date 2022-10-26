import argparse
import os
import time

from diffusers import OnnxStableDiffusionPipeline


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
    from torch import float16
    from diffusers import StableDiffusionPipeline

    if precision == "fp16":
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", torch_dtype=float16, revision=precision, use_auth_token=True
        ).to("cuda")
        pipe.unet_jit = unet_jit
    else:
        # pipe = StableDiffusionPipeline.from_pretrained(
        #     "CompVis/stable-diffusion-v1-4", use_auth_token=True
        # ).to("cuda")
        print("Skipping PyTorch FP32 for now")
        exit(1)
    return pipe


def run_pipeline(pipe):
    # Warm up
    height = 512
    width = 512
    pipe("warm up", height, width, num_inference_steps=2)

    # Test inputs
    prompts = [  
        # "a photo of an astronaut riding a horse on mars",
        "cute grey cat with blue eyes, wearing a bowtie, acrylic painting"
    ]

    num_inference_steps = 50
    start = time.time()
    pipe(prompts, height, width, num_inference_steps)
    end = time.time()
    print(f"Inference took {end - start} seconds")


# Measure throughput at the maximum batch size
def measure_throughput(pipe):
    # Warm up
    height, width = 512, 512
    pipe("warm up", height, width, num_inference_steps=2)

    num_inference_steps = 50
    min_batch_size, max_batch_size = 1, 1024
    
    # Search for maximum batch size
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
            pipe(prompts, height, width, num_inference_steps=num_inference_steps)
            end = time.time()
            print(f"Batch size = {batch_size}, latency = {end - start} s, throughput = {(num_inference_steps * batch_size) / (end - start)} it/s")
            
            print(f"Batch size = {batch_size} is too low. Refining search space for min batch size.")
            if isinstance(pipe, OnnxStableDiffusionPipeline):
                min_batch_size += 1
            else:
                min_batch_size = batch_size+1
        except:
            print(f"Batch size = {batch_size} is too high. Refining search space for max batch size.")
            max_batch_size = batch_size-1

    print(f"Search is complete. Max batch size = {max_batch_size}.")


def profile_memory(pipe, max_batch_size, height=512, width=512, num_inference_steps=50):
    from onnxruntime.transformers.benchmark_helper import measure_memory
    from torch.cuda import empty_cache
    from gc import collect

    # Garbage collect before measuring
    collect()
    empty_cache()

    print("Measuring memory usage at batch size = 1:")
    prompts = ["a photo of an astronaut riding a horse on mars"]
    measure_memory(is_gpu=True, func=lambda: pipe(prompts, height, width, num_inference_steps=num_inference_steps))

    # Garbage collect before measuring
    collect()
    empty_cache()

    print(f"Measuring memory usage at max batch size = {max_batch_size}:")
    prompts = ["a photo of an astronaut riding a horse on mars" for _ in range(max_batch_size)]
    measure_memory(is_gpu=True, func=lambda: pipe(prompts, height, width, num_inference_steps=num_inference_steps))


def choose_metric(pipe, metric, max_batch_size):
    assert metric in {"latency", "throughput", "memory"}

    if metric == "latency":
        if isinstance(pipe, OnnxStableDiffusionPipeline):
            run_pipeline(pipe)
        else:
            with torch.autocast("cuda"):
                run_pipeline(pipe)
    
    elif metric == "throughput":
        measure_throughput(pipe)

    else:
        assert max_batch_size > 0
        profile_memory(pipe, max_batch_size)


def run_ort(directory, provider, metric, max_batch_size):
    load_start = time.time()
    pipe = get_ort_pipeline(directory, provider)
    load_end = time.time()
    print(f"Model loading took {load_end - load_start} seconds")
    choose_metric(pipe, metric, max_batch_size)


def run_torch(disable_conv_algo_search, precision, unet_jit, metric, max_batch_size):
    import torch

    if not disable_conv_algo_search:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    load_start = time.time()
    pipe = get_torch_pipeline(precision, unet_jit)
    load_end = time.time()
    print(f"Model loading took {load_end - load_start} seconds")
    choose_metric(pipe, metric, max_batch_size)


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
        "--max_batch_size",
        required=False,
        type=int,
        default=0,
        help="Maximum batch size found during search when measuring throughput (required when metric is 'memory')",
    )

    parser.add_argument(
        "-m",
        "--metric",
        required=True,
        type=str,
        help="Metric to evaluate pipeline on",
        choices=["latency", "throughput", "memory"]
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
        run_ort(args.pipeline, provider, args.metric, args.max_batch_size)
    else:
        run_torch(args.disable_conv_algo_search, args.floating_point_precision, args.unet_jit, args.metric, args.max_batch_size)


if __name__ == "__main__":
    main()
