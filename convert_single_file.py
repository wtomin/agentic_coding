import argparse
import os
import shutil


from convert_folder import convert_file, TorchToMindsporeCST



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    shutil.copy(args.input_file, args.output_file)
    convert_file(args.output_file, TorchToMindsporeCST)
    print(f"[+] Converted {args.input_file} to {args.output_file}")


if __name__ == "__main__":
    main()