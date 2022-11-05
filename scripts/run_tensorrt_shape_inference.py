import argparse
import os
import shutil
import subprocess


def infer_shapes(output_folder, verbose):
    for root, dirs, files in os.walk(output_folder):
        for fle in files:
            if ".onnx" in fle:
                path = os.path.join(root, fle)
                # cmd = ['python3', '-m', 'onnxruntime.tools.symbolic_shape_infer', '--input', path, '--output', path, '--verbose', str(verbose)]
                cmd = ['python3', 'scripts/symbolic_shape_infer.py', '--input', path, '--output', path, '--verbose', str(verbose)]
                if "unet" in fle:
                    cmd.append('--save_as_external_data')
                if "unet" in fle or "text_encoder" in fle:
                    cmd.append('--auto_merge')

                print(f"Running ORT symbolic shape inference script for {path}")
                subprocess.run(cmd)
                print(f"Finished ORT symbolic shape inference script for {path}")


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
    parser.add_argument(
        '-v',
        '--verbose',
        required=False,
        type=int,
        default=0,
        choices=[0, 1, 3],
        help='Verbosity level of output logs (0 for none, 1 for warnings, 3 for all)',
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
    infer_shapes(args.output, args.verbose)


if __name__ == '__main__':
    main()
