################################################################################################
# Convert stable diffusion models in ONNX from FP32 to FP16
#
# Notes: 
# 1) This script needs to be run specifically with ORT 1.13 in order to work.
# 2) This script needs to be located in the same folder as the parent folder to the FP32 files.
################################################################################################

import argparse
import os
import shutil
import onnx
from onnxruntime.transformers.optimizer import optimize_model

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', required=True, 
                    help="Root directory of the ONNX pipeline data files (ex: ./sd_onnx_v4). \
                          Must be located in the same folder as the parent folder to the FP32 files.")
args = parser.parse_args()

for name in ["unet", "vae_encoder", "vae_decoder", "text_encoder", "safety_checker"]:
    onnx_model_path = f"{args.root_dir}/{name}/model.onnx"

    # The following will fuse LayerNormalization and Gelu. Do it before fp16 conversion, otherwise they cannot be fused later.
    # Right now, onnxruntime does not save >2GB model so we use script to optimize unet instead.
    m = optimize_model(
        onnx_model_path,
        model_type="bert",
        num_heads=0,
        hidden_size=0,
        opt_level=0,
        optimization_options=None,
        use_gpu=False,
    )

    # Use op_block_list to force some operators to compute in FP32.
    # TODO: might need some tuning to add more operators to op_block_list to reduce accuracy loss.
    if name == "safety_checker":
        m.convert_float_to_float16(op_block_list=["Where"])
    else:
        m.convert_float_to_float16()

    # Overwrite existing models. You can change it to another directory but need copy other files like tokenizer manually.
    optimized_model_path = f"{args.root_dir}/{name}/model.onnx"
    output_dir = os.path.dirname(optimized_model_path)
    shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    onnx.save_model(m.model, optimized_model_path)