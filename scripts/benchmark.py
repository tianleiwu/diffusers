import argparse
import gc
import os
import time
import torch

from diffusers import OnnxStableDiffusionPipeline, StableDiffusionPipeline

MODEL_NAME = "CompVis/stable-diffusion-v1-4"

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
        MODEL_NAME,
        revision="onnx",
        provider=provider,
        use_auth_token=True,
    )
    return pipe


def get_torch_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME, 
        revision="fp16",
        torch_dtype=torch.float16, 
        use_auth_token=True,
    ).to("cuda")
    return pipe


def get_torchscript_pipeline(disable_channels_last):
    trace_unet(disable_channels_last)
    filename = get_trace_file(disable_channels_last)

    from dataclasses import dataclass
    @dataclass
    class UNet2DConditionOutput:
        sample: torch.FloatTensor

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=True,
    ).to("cuda")

    # use jitted unet
    unet_traced = torch.jit.load(filename)
    # del pipe.unet
    class TracedUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.in_channels = pipe.unet.in_channels
            self.device = pipe.unet.device

        def forward(self, latent_model_input, t, encoder_hidden_states):
            sample = unet_traced(latent_model_input, t, encoder_hidden_states)[0]
            return UNet2DConditionOutput(sample=sample)

    pipe.unet = TracedUNet()
    print(pipe.scheduler)
    return pipe


def get_trace_file(disable_channels_last):
    return "unet_traced_channels_last.pt" if not disable_channels_last else "unet_traced.pt"


def trace_unet(disable_channels_last):
    filename = get_trace_file(disable_channels_last)
    if os.path.exists(filename):
        print("Skip tracing since file exists.")
        return

    import functools

    # torch disable grad
    torch.set_grad_enabled(False)

    # set variables
    n_experiments = 2
    unet_runs_per_experiment = 50

    # load inputs
    def generate_inputs():
        sample = torch.randn(2, 4, 64, 64).half().cuda()
        timestep = torch.rand(1).half().cuda() * 999
        encoder_hidden_states = torch.randn(2, 77, 768).half().cuda()
        return sample, timestep, encoder_hidden_states

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=True,
    ).to("cuda")

    unet = pipe.unet
    unet.eval()
    if not disable_channels_last:
        unet.to(memory_format=torch.channels_last) # use channels_last memory format
    unet.forward = functools.partial(unet.forward, return_dict=False) # set return_dict=False as default

    # warmup
    for _ in range(5):
        with torch.inference_mode():
            inputs = generate_inputs()
            orig_output = unet(*inputs)

    # trace
    print("Tracing...")
    unet_traced = torch.jit.trace(unet, inputs)
    unet_traced.eval()
    print("Done tracing")

    # warmup and optimize graph
    for _ in range(5):
        with torch.inference_mode():
            inputs = generate_inputs()
            orig_output = unet_traced(*inputs)

    with torch.inference_mode():
        for _ in range(n_experiments):
            torch.cuda.synchronize()
            start_time = time.time()
            for _ in range(unet_runs_per_experiment):
                orig_output = unet_traced(*inputs)
            torch.cuda.synchronize()
            print(f"Unet traced inference took {time.time() - start_time:.2f} seconds")
        for _ in range(n_experiments):
            torch.cuda.synchronize()
            start_time = time.time()
            for _ in range(unet_runs_per_experiment):
                orig_output = unet(*inputs)
            torch.cuda.synchronize()
            print(f"Unet inference took {time.time() - start_time:.2f} seconds")

    # save the model
    unet_traced.save(filename)


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


def run_torch(disable_conv_algo_search, mode, batch_size):
    if not disable_conv_algo_search:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    # torch disable grad
    torch.set_grad_enabled(False)
    
    load_start = time.time()
    pipe = get_torch_pipeline()
    load_end = time.time()
    print(f"Model loading took {load_end - load_start} seconds")
    choose_mode(pipe, mode, batch_size)


def run_torchscript(disable_channels_last, mode, batch_size):
    load_start = time.time()
    pipe = get_torchscript_pipeline(disable_channels_last)
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
        choices=["onnxruntime", "torch", "torchscript"],
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
        "-ep",
        "--provider",
        required=False,
        type=str,
        default="cuda",
        choices=["cuda", "trt", "cpu"],
        help="Execution provider of ONNX pipeline"
    )

    parser.add_argument(
        "-a",
        "--disable_conv_algo_search",
        required=False,
        action="store_true",
        help="Disable cuDNN conv algo search.",
    )
    parser.set_defaults(disable_conv_algo_search=False)

    parser.add_argument(
        "-c",
        "--disable_channels_last",
        required=False,
        action="store_true",
        help="Disable channels last (for TorchScript)",
    )
    parser.set_defaults(disable_channels_last=False)

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
    print(args.__dict__)

    if args.engine == "onnxruntime":
        providers = {"cuda": "CUDAExecutionProvider", "trt": "TensorrtExecutionProvider", "cpu": "CPUExecutionProvider"}
        args.provider = providers[args.provider]
        if args.provider == "CUDAExecutionProvider" and args.disable_conv_algo_search:
            args.provider = [
                (
                    args.provider,
                    {"cudnn_conv_use_max_workspace": "1", "cudnn_conv_algo_search": "EXHAUSTIVE"},
                ),
            ]
        else:
            args.provider = [args.provider]
        run_ort(args.pipeline, args.provider, args.mode, args.batch_size)
    elif args.engine == "torch":
        run_torch(args.disable_conv_algo_search, args.mode, args.batch_size)
    else:
        run_torchscript(args.disable_channels_last, args.mode, args.batch_size)


if __name__ == "__main__":
    main()
