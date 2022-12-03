# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# Before running this script, you need convert checkpoint to float32 onnx models like the following
#    git clone https://github.com/huggingface/diffusers
#    cd diffusers
#    pip install -e .
#    huggingface-cli login
#    python3 scripts/convert_stable_diffusion_checkpoint_to_onnx.py --model_path runwayml/stable-diffusion-v1-5  --output_path ../stable-diffusion-v1-5
#
# Then you can use this script to convert them to float16 like the following:
#    pip3 install -U onnxruntime-gpu
#    python3 scripts/convert_sd_onnx_to_fp16.py -i ../stable-diffusion-v1-5 -o ../stable-diffusion-v1-5-fp16

import argparse
import os
import shutil
from pathlib import Path

from onnxruntime.transformers.optimizer import optimize_model


def convert_to_fp16(source_dir: Path, target_dir: Path, overwrite: bool, use_external_data_format: bool):
    dirs_with_onnx = ["vae_encoder", "vae_decoder", "text_encoder", "safety_checker", "unet"]
    for name in dirs_with_onnx:
        onnx_model_path = source_dir / name / "model.onnx"

        if not os.path.exists(onnx_model_path):
            raise RuntimeError(f"onnx model does not exist: {onnx_model_path}")

        # The following will fuse LayerNormalization and Gelu.
        # Do it before fp16 conversion, otherwise they cannot be fused later.
        # Right now, onnxruntime does not save >2GB model so we use script to optimize unet instead.
        m = optimize_model(
            str(onnx_model_path),
            model_type="bert",
            num_heads=0,
            hidden_size=0,
            opt_level=0,
            optimization_options=None,
            use_gpu=False,
        )

        m.convert_float_to_float16(op_block_list=["RandomNormalLike", "Resize"])

        optimized_model_path = target_dir / name / "model.onnx"
        output_dir = optimized_model_path.parent
        if optimized_model_path.exists():
            if not overwrite:
                raise RuntimeError(f"output onnx model path existed: {optimized_model_path}")

        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        m.save_model_to_file(str(optimized_model_path), use_external_data_format=use_external_data_format)
        print(f"{onnx_model_path} => {optimized_model_path}")


def copy_extra(source_dir: Path, target_dir: Path, overwrite: bool):
    extra_dirs = ["scheduler", "tokenizer", "feature_extractor"]
    for name in extra_dirs:
        source_path = source_dir / name
        if not os.path.exists(source_path):
            raise RuntimeError(f"source path does not exist: {source_path}")

        target_path = target_dir / name
        if target_path.exists():
            if not overwrite:
                raise RuntimeError(f"output path existed: {target_path}")
            shutil.rmtree(target_path)

        shutil.copytree(source_path, target_path)
        print(f"{source_path} => {target_path}")

    extra_files = ["model_index.json"]
    for name in extra_files:
        source_path = source_dir / name
        if not os.path.exists(source_path):
            raise RuntimeError(f"source path does not exist: {source_path}")

        target_path = target_dir / name
        if target_path.exists():
            if not overwrite:
                raise RuntimeError(f"output path existed: {target_path}")
            os.remove(target_path)
        shutil.copyfile(source_path, target_path)
        print(f"{source_path} => {target_path}")


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help="Root of input directory of stable diffusion onnx pipeline with float32 models.",
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
        help="Root of output directory of stable diffusion onnx pipeline with float16 models.",
    )

    parser.add_argument(
        "--overwrite",
        required=False,
        action="store_true",
        help="Overwrite exists files.",
    )
    parser.set_defaults(overwrite=False)

    parser.add_argument(
        "-e",
        "--use_external_data_format",
        required=False,
        action="store_true",
        help="Onnx model larger than 2GB need to use external data format.",
    )
    parser.set_defaults(use_external_data_format=False)

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    copy_extra(Path(args.input), Path(args.output), args.overwrite)
    convert_to_fp16(Path(args.input), Path(args.output), args.overwrite, args.use_external_data_format)


main()
