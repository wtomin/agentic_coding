# GOALS

Given the input folders of pytorch python scripts (or some python scripts have been partially converted to mindspore), this tool should:
1. Automatically find all python scripts that need conversion (Pytorch -> MindSpore). 
2. Automatically convert all pytorch scripts to mindspore.
3. Keep the output folder structure same as the input folder structure.


# Input Processing 

### PHASE 1: Fast File Detection (Automated)

```python
def scan_and_categorize_inputs():
    """
    Discover all PyTorch files for conversion
    Returns: list of relative file paths
    """
    files = get_torch_files_to_convert('inputs/')
    return sorted(files['convert'])  # Deterministic order
```

`get_torch_files_to_convert` uses `libcst` to find all files that contain torch-related imports, for example:
1. torch
2. torchvision
3. transformers/diffusers
4. timm
5. pytorch_lightning
6. accelerate
7. torchmetrics
8. vllm/sglang
9. other libraries that use torch


### PHASE 2: Automatic Conversion with Diff Review

For each file to be converted, refer to the rules in `CLAUDE.md` to convert the file. Always create a backup before writing, and ask for human to review the diff.

During conversion, if the pytorch file is one of modeling, inference, training, and dataset scripts, please read the examples in `examples/modeling`, `examples/inference`, `examples/train`, and `examples/dataset` to find the reference code.


### PHASE 3: Structure Replication

**GUARANTEE:** Every converted file maintains exact relative path
```
inputs/subfolder/model.py → outputs/subfolder/model.py
inputs/utils/data.py → outputs/utils/data.py
```

## Critical Components

### Folder Structure Automation
```python
auto_convert.py:
- [ ] auto_scan() - return list[file_paths]
- [ ] auto_convert() - apply to all files
- [ ] create_diffs() - generate .diff files
- [ ] backup_system() - handle .backup files
- [ ] structure_mirror() - guarantee path preservation
```

### Human Review Integration
```python
# After conversion, human reviews:
# file.outputs/subfolder/model.py.diff
# file.outputs/subfolder/model.py.backup (original)
# Human can: edit outputs/subfolder/model.py directly or reject
```


### Examples

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

## Workflow


1. Process input files: process all input files  → idenfity files to convert  → simply copy those files that are not to be converted to outputs
2. Start automatic conversion: for each torch file, first save backup, then generate diffs → human reviews diffs → can modify outputs afterward
3. Review final outputs folder, if any file is not converted, ask for manual review