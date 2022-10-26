import argparse
import os
import time

from diffusers import OnnxStableDiffusionPipeline

MODEL_NAME = "CompVis/stable-diffusion-v1-4" #"runwayml/stable-diffusion-v1-5"

def get_ort_pipeline(directory, provider):
    import onnxruntime

    if directory is not None:
        assert os.path.exists(directory)
        session_options = onnxruntime.SessionOptions()
        pipe = OnnxStableDiffusionPipeline.from_pretrained(
            directory,
            provider=provider,
            sess_options=session_options,
        )
        return pipe

    # Original FP32 ONNX models
    pipe = OnnxStableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        revision="onnx",
        provider=provider,
        use_auth_token=True,
    )

    print(pipe.scheduler)
    return pipe


def get_torch_pipeline(disable_channels_last):
    from torch import float16, channels_last
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME, torch_dtype=float16, revision="fp16", use_auth_token=True
    ).to("cuda")

    #pipe.enable_attention_slicing()

    if not disable_channels_last:
        pipe.unet.to(memory_format=channels_last)  # in-place operation

    return pipe

def get_trace_file(disable_channels_last):
    return "unet_traced_channels_last.pt" if not disable_channels_last else "unet_traced.pt"

def trace_unet(disable_channels_last):
    filename = get_trace_file(disable_channels_last)
    if not os.path.exists(filename):
        print("skip tracing since file existed.")
        return

    import time
    import torch
    from diffusers import StableDiffusionPipeline
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
    ).to("cuda")
    unet = pipe.unet
    unet.eval()
    if not disable_channels_last:
        unet.to(memory_format=torch.channels_last)  # use channels_last memory format
    unet.forward = functools.partial(unet.forward, return_dict=False)  # set return_dict=False as default

    # warmup
    for _ in range(3):
        with torch.inference_mode():
            inputs = generate_inputs()
            orig_output = unet(*inputs)

    # trace
    print("tracing..")
    unet_traced = torch.jit.trace(unet, inputs)
    unet_traced.eval()
    print("done tracing")


    # warmup and optimize graph
    for _ in range(5):
        with torch.inference_mode():
            inputs = generate_inputs()
            orig_output = unet_traced(*inputs)


    # benchmarking
    with torch.inference_mode():
        for _ in range(n_experiments):
            torch.cuda.synchronize()
            start_time = time.time()
            for _ in range(unet_runs_per_experiment):
                orig_output = unet_traced(*inputs)
            torch.cuda.synchronize()
            print(f"unet traced inference took {time.time() - start_time:.2f} seconds")
        for _ in range(n_experiments):
            torch.cuda.synchronize()
            start_time = time.time()
            for _ in range(unet_runs_per_experiment):
                orig_output = unet(*inputs)
            torch.cuda.synchronize()
            print(f"unet inference took {time.time() - start_time:.2f} seconds")

    # save the model
    unet_traced.save(filename)

def load_traced(disable_channels_last):
    filename = get_trace_file(disable_channels_last)

    from diffusers import StableDiffusionPipeline
    import torch
    from dataclasses import dataclass

    @dataclass
    class UNet2DConditionOutput:
        sample: torch.FloatTensor

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        revision="fp16",
        torch_dtype=torch.float16,
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

def get_torchscript_pipeline(disable_channels_last):
    trace_unet(disable_channels_last)
    return load_traced(disable_channels_last)

def run_ort_pipeline(pipe, batch_size):
    assert isinstance(pipe, OnnxStableDiffusionPipeline)

    # Warm up
    height = 512
    width = 512
    pipe("warm up", height, width, num_inference_steps=2)

    # Test inputs
    prompts = [
        "a photo of an astronaut riding a horse on mars",
        #"cute grey cat with blue eyes, wearing a bowtie, acrylic painting"
    ]

    num_inference_steps = 50
    for i, prompt in enumerate(prompts):
        input_prompts = [prompt] * batch_size
        inference_start = time.time()
        image = pipe(input_prompts, height, width, num_inference_steps).images[0]
        inference_end = time.time()

        print(f"Inference took {inference_end - inference_start} seconds")
        image.save(f"onnx_{i}.jpg")

def run_torch_pipeline(pipe, batch_size):
    import torch
    # Warm up
    height = 512
    width = 512
    pipe("warm up", height, width, num_inference_steps=2)

    # Test inputs
    prompts = [
        "a photo of an astronaut riding a horse on mars",
        #"cute grey cat with blue eyes, wearing a bowtie, acrylic painting"
    ]

    # torch disable grad
    torch.set_grad_enabled(False)

    num_inference_steps = 50
    for i, prompt in enumerate(prompts):
        input_prompts = [prompt] * batch_size
        torch.cuda.synchronize()
        inference_start = time.time()
        image = pipe(input_prompts, height, width, num_inference_steps).images[0]
        torch.cuda.synchronize()
        inference_end = time.time()

        print(f"Inference took {inference_end - inference_start} seconds")
        image.save(f"torch_{i}.jpg")


def run_ort(directory, provider, batch_size):
    load_start = time.time()
    pipe = get_ort_pipeline(directory, provider)
    load_end = time.time()
    print(f"Model loading took {load_end - load_start} seconds")
    run_ort_pipeline(pipe, batch_size)

def run_torch(disable_conv_algo_search, batch_size, disable_channels_last, torchscript=True):
    import torch

    if not disable_conv_algo_search:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    #torch.backends.cuda.matmul.allow_tf32 = True

    # torch disable grad
    torch.set_grad_enabled(False)
    
    load_start = time.time()
    if torchscript:
        pipe = get_torchscript_pipeline(disable_channels_last)
    else:
        pipe = get_torch_pipeline(disable_channels_last)
    load_end = time.time()
    print(f"Model loading took {load_end - load_start} seconds")

    with torch.inference_mode():
        run_torch_pipeline(pipe, batch_size)


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
        help="Disable channels_last.",
    )
    parser.set_defaults(disable_channels_last=False)


    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=1
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    print(args)
    
    if args.engine == "onnxruntime":
        provider = (
            ["CUDAExecutionProvider"]
            if args.disable_conv_algo_search
            else [
                (
                    "CUDAExecutionProvider",
                    {"cudnn_conv_use_max_workspace": "1", "cudnn_conv_algo_search": "EXHAUSTIVE"},
                ),
                # "CPUExecutionProvider",
            ]
        )
        run_ort(args.pipeline, provider, args.batch_size)
    else:
        run_torch(args.disable_conv_algo_search, args.batch_size, args.disable_channels_last, args.engine == "torchscript")

if __name__ == "__main__":
    main()
