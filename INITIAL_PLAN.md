# GOALS

Given the input folders of pytorch python scripts (or some python scripts have been partially converted to mindspore), this tool should:
1. Automatically find all python scripts that need conversion (Pytorch -> MindSpore), keep a track note of files to be converted named as "files_to_convert.md". 
2. Automatically convert all pytorch scripts to mindspore. After one file is converted, you can cross-check the coverted file in "files_to_convert.md". 
3. Complete the conversion and give a report, save the report in "report.md".


# Input Processing 

### PHASE 1: Fast File Detection (Automated)

Use `libcst` or regular match to find all files that contain torch-related codes, for example:
1. torch
2. torchvision
3. transformers/diffusers
4. timm
5. pytorch_lightning
6. accelerate
7. torchmetrics
8. vllm/sglang
9. other libraries that use torch

Save all files that contain torch-related codes into a to-do list, saved them to "files_to_convert.md".

"files_to_convert.md" contains all files paths that need to be converted, recorded in a section named "Files". You should also create another section named "Edits" to record all edits to be made to each file.

### PHASE 2: Automatic Conversion with Diff Review

#### Examples references:

Read the examples scripts in `examples/`, and make a short summary on the conversion rules for modeling, data, training, and inference.

Read the files in "files_to_convert.md" one by one. Firstly, categorize the files into one of the following categories:
1. Modeling
2. Data
3. Training
4. Inference

For each file, recall the examples in corresponding categories before conversion.


In `examples/` folder, there are some examples of python script conversion:
1. `examples/modeling`:
- `modeling_cohere2_torch.py`: Cohere2 model's torch script;
- `modeling_cohere2.py`: Cohere2 model's MindSpore script;
- `test_modeling_cohere2.py`: Test script for the Cohere2 model, which aims to compare the precision error between MindSpore and PyTorch.
- `configuration_cohere2.py`: the configuration file for Cohere2 model, shared by both PyTorch and MindSpore.

2. `examples/inference`:
- `generate_torch.py`: an example of PyTorch inference script with Cohere2 Model;
- `generate_ms.py`: an example of MindSpore inference script with Cohere2 Model.

3. `examples/dataset`:
- `README.md`: the README file for the difference between PyTorch and MindSpore in dataset definition;
- `dataset_torch.py`: an example dataset definition for PyTorch;
- `dataset.py`: an example dataset definition for MindSpore.

4. `examples/train`:
- `REAMDE.md`: the README file for the difference between PyTorch and MindSpore in training script;

#### Conversion 

For each file to be converted, refer to the rules in `CLAUDE.md` to convert the file. Always ask for human to review the diff.

In `outputs` folder, if a file with the same name exists, it means the file is partially converted using AST rules. You need to further convert it with the rules in `CLAUDE.md`.

After the conversion is done, save the edits you made to the section named Edits in files_to_convert.md. Continue to read files_to_convert.md to the next file.

### PHASE 3: Complete

- Check the output files, and make sure they are correctly converted. No torch-related code should be left in the output files.
- If there are errors, fix them and re-run the process.
- If no errors, you should generate a conversion report based on the files_to_convert.md. Save the report to a file named conversion_report.md.



## Workflow

1. Process input files: process all input files  → idenfity files to convert  → save files paths to a file named files_to_convert.md
2. Start automatic conversion: for each torch file, generate diffs → human reviews diffs → can modify outputs afterward  → update files_to_convert.md
3. Review final outputs folder, if any file is not converted.
4. Generate a report of all files converted, and all files not converted, based on files_to_convert.md.