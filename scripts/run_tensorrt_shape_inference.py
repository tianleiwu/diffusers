import argparse
import os
import shutil
import subprocess

import onnx
from onnxruntime.transformers.shape_infer_helper import SymbolicShapeInferenceHelper


def infer_shapes(output_folder):
    for root, dirs, files in os.walk(output_folder):
        for fle in files:
            if ".onnx" in fle:
                path = os.path.join(root, fle)
                cmd = ['python3', '-m', 'onnxruntime.tools.symbolic_shape_infer', 
                       '--input', path, '--output', path, '--auto_merge']
                if "unet" in path:
                    cmd.append('--save_as_external_data')
                
                print(f"Running symbolic shape inference for {path}")

                # Option 1: ORT symbolic shape infer script
                print(" ".join(cmd))
                subprocess.run(cmd)
                
                # Option 2: SymbolicShapeInferenceHelper
                # helper = SymbolicShapeInferenceHelper(onnx.load(path), auto_merge=False)
                # dynamic_axis_mapping = {'batch': 1, 'batch_size': 1} # Add additional mappings as needed
                # helper.infer(dynamic_axis_mapping)

                # Option 3: ONNX symbolic shape infer script
                # if "unet" in path:
                #     onnx.shape_inference.infer_shapes_path(path)
                # else:
                #     model = onnx.load(path)
                #     onnx.checker.check_model(model)
                #     model_new = onnx.shape_inference.infer_shapes(model)
                #     onnx.checker.check_model(model_new)
                #     onnx.save(model_new, path)

                print(f"Finished symbolic shape inference for {path}", end='\n\n')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        required=True,
        type=str,
        help='Input path to folder of ONNX models',
    )
    parser.add_argument(
        '-o',
        '--output',
        required=False,
        type=str,
        default='',
        help='Output path to store shape-inferred ONNX models',
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    if args.output != '':
        if os.path.exists(args.output):
            print(f"WARNING: Removing existing directory '{args.output}'")
            shutil.rmtree(args.output)
        shutil.copytree(args.input, args.output)
    else:
        args.output = args.input
    infer_shapes(args.output)


if __name__ == '__main__':
    main()
