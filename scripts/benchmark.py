import time
import argparse
import os

def get_ort_pipeline(directory):
    import onnxruntime
    from diffusers import StableDiffusionOnnxPipeline

    if directory is not None:
        assert os.path.exists(directory)
        session_options = onnxruntime.SessionOptions()
        pipe = StableDiffusionOnnxPipeline.from_pretrained(
            directory,
            provider="CUDAExecutionProvider",
            session_options=session_options,
        )
        return pipe
    
    #Original FP32 ONNX models
    pipe = StableDiffusionOnnxPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        revision="onnx",
        provider="CUDAExecutionProvider",
        use_auth_token=True,
    )
    return pipe

def get_torch_pipeline():
    from diffusers import StableDiffusionPipeline

    from torch import float16
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", 
        torch_dtype=float16,
        revision="fp16",
        use_auth_token=True).to("cuda")
    return pipe

def run_pipeline(pipe):
    # Warm up
    height = 512
    width = 512
    pipe("warm up", height, width, num_inference_steps=2)

    # Test inputs
    prompts = ["a photo of an astronaut riding a horse on mars",
            "cute grey cat with blue eyes, wearing a bowtie, acrylic painting"]

    num_inference_steps = 50
    for i, prompt in enumerate(prompts):
        inference_start = time.time()
        image = pipe(prompt, height, width, num_inference_steps).images[0]
        inference_end = time.time()

        print(f'Inference took {inference_end - inference_start} seconds')
        image.save("onnx_{i}.jpg")

def run_ort(directory):
    load_start = time.time()
    pipe = get_ort_pipeline(directory)
    load_end = time.time()
    print(f'Model loading took {load_end - load_start} seconds')
    run_pipeline(pipe)

def run_torch():
    load_start = time.time()
    pipe = get_torch_pipeline()
    load_end = time.time()
    print(f'Model loading took {load_end - load_start} seconds')

    from torch import autocast
    with autocast("cuda"):
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
        "--provider",
        required=False,
        type=str,
        default="CUDAExecutionProvider",
        help="Execution provider to use",
    )

    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    if args.engine == "onnxruntime":
        run_ort(args.pipeline)
    else:
        run_torch()

if __name__ == "__main__":
    main()
