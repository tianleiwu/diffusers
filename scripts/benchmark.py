import argparse
import os
import time

from diffusers import StableDiffusionOnnxPipeline


def get_ort_pipeline(directory, provider):
    import onnxruntime

    if directory is not None:
        assert os.path.exists(directory)
        session_options = onnxruntime.SessionOptions()
        pipe = StableDiffusionOnnxPipeline.from_pretrained(
            directory,
            provider=provider,
            session_options=session_options,
        )
        return pipe

    # Original FP32 ONNX models
    pipe = StableDiffusionOnnxPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        revision="onnx",
        provider=provider,
        use_auth_token=True,
    )
    return pipe


def get_torch_pipeline():
    from torch import float16

    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", torch_dtype=float16, revision="fp16", use_auth_token=True
    ).to("cuda")
    return pipe


def run_pipeline(pipe):
    # Warm up
    height = 512
    width = 512
    
    def warmup():
        pipe("warm up", height, width, num_inference_steps=2)

    from onnxruntime.transformers.benchmark_helper import measure_memory
    measure_memory(True, warmup)
    
    # Test inputs
    prompts = [  # "a photo of an astronaut riding a horse on mars",
        "cute grey cat with blue eyes, wearing a bowtie, acrylic painting"
    ]

    num_inference_steps = 50
    for i, prompt in enumerate(prompts):
        inference_start = time.time()
        image = pipe(prompt, height, width, num_inference_steps).images[0]
        inference_end = time.time()

        print(f"Inference took {inference_end - inference_start} seconds")
        if isinstance(pipe, StableDiffusionOnnxPipeline):
            image.save(f"onnx_{i}.jpg")
        else:
            image.save(f"torch_{i}.jpg")


def run_ort(directory, provider):
    load_start = time.time()
    pipe = get_ort_pipeline(directory, provider)
    load_end = time.time()
    print(f"Model loading took {load_end - load_start} seconds")
    run_pipeline(pipe)


def run_torch(disable_conv_algo_search):
    import torch

    if not disable_conv_algo_search:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    load_start = time.time()
    pipe = get_torch_pipeline()
    load_end = time.time()
    print(f"Model loading took {load_end - load_start} seconds")

    with torch.autocast("cuda"):
        run_pipeline(pipe)


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
        help="Disable cuDNN conv algo search. ",
    )
    parser.set_defaults(disable_conv_algo_search=False)

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    if args.engine == "onnxruntime":
        #assert "CUDAExecutionProvider" in onnxruntime.get_available_providers()
        cuda_provider_settings = {
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_use_max_workspace": "1",
            #"cudnn_conv_algo_search": "EXHAUSTIVE",
            #"cudnn_conv1d_pad_to_nc1d": True,
            #"do_copy_in_default_stream": False
        }
        cuda_providers = ["CUDAExecutionProvider"] if args.disable_conv_algo_search else [
                ("CUDAExecutionProvider", cuda_provider_settings),
                #"CPUExecutionProvider",
            ]

        providers = {
            "unet": cuda_providers,
            'safety_checker':cuda_providers,
            'text_encoder':cuda_providers,
            'vae_decoder':cuda_providers,
            'vae_encoder':cuda_providers
        }

        run_ort(args.pipeline, provider)
    else:
        run_torch(args.disable_conv_algo_search)


if __name__ == "__main__":
    main()
