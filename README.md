# Agentic Code Conversion Tool

An agentic tool for automated conversion of PyTorch code to MindSpore, allowing various inputs.

## Overview

This project provides a comprehensive solution for converting PyTorch-based machine learning code to MindSpore, with a focus on maintaining code functionality while adapting to MindSpore's implementation. The tool uses a combination of AST (Abstract Syntax Tree) transformation and pattern matching to ensure accurate conversions.

## Quick Start

Firstly, please prepare a torch repository to be converted:
```bash
git clone https://github.com/wtomin/agentic_coding
git checkout flex-inputs
cd agentic_coding
cd example_inputs 
bash download.sh  # download an example torch repository https://github.com/ivanwhaf/yolov1-pytorch/tree/master
cd ..
mv example_inputs/ inputs/
```

Then you can generate task files with:
```bash
python task_generator.py
```
Here is the output:
```bash
Starting task generation for directory: inputs
Scanning 10 Python files...
[-] Skipped inputs/yolov1-pytorch/detect.py (no PyTorch usage detected)
[+] Created task for inputs\yolov1-pytorch\train.py (category: training)
[+] Created task for inputs\yolov1-pytorch\models\yolov1.py (category: modeling)
[-] Skipped inputs/yolov1-pytorch/models/__init__.py (no PyTorch usage detected)
[+] Created task for inputs\yolov1-pytorch\utils\datasets.py (category: dataset)
[-] Skipped inputs/yolov1-pytorch/utils/draw_gt.py (no PyTorch usage detected)
[+] Created task for inputs\yolov1-pytorch\utils\loss.py (category: training)
[-] Skipped inputs/yolov1-pytorch/utils/modify_label.py (no PyTorch usage detected)
[+] Created task for inputs\yolov1-pytorch\utils\util.py (category: others)
[-] Skipped inputs/yolov1-pytorch/utils/__init__.py (no PyTorch usage detected)

=== Task Generation Summary ===
Total tasks generated: 5
Categories: {'modeling': 1, 'dataset': 1, 'training': 2, 'others': 1}
Priority distribution: {'1': 1, '3': 1, '4': 2, '5': 1}

[+] Generated 5 conversion tasks
[+] Task files saved to: tasks
```

In the `tasks/` folder, there are multiple task file with `.json` extension, which corresponds to the python script to be converted.

Finally, you can start the conversion with the coding agent, like:
```text
The current task file is @tasks/task_001.json. Read @CONVERT.md and start the conversion task.
```
This will convert the python file in `tasks/task_001.json`. We recommend to convert the files in a parallel and independent manner, especially when the number of tasks files is large.

If the number of tasks files is small, you may consider asking the coding agent to convert all files listed in `tasks\task_summary.json`. The performance maybe dergaded when the context length is too long.


## Project Structure

```
agentic_coding/
├── task_generator.py        # Main task generation orchestrator
├── convert_folder.py        # Bulk conversion implementation
├── convert_single_file.py   # Single file conversion utility
├── examples/               # Example code and conversion references
│   ├── dataset/           # Dataset-related examples
│   ├── inference/         # Inference code examples
│   ├── modeling/         # Model architecture examples
│   └── training/         # Training script examples
├── example_inputs/       # Directory for sample PyTorch files
├── inputs/                # Directory for input PyTorch files
└── outputs/              # Directory for converted MindSpore files
```

## Documentation

### Task Generator (`task_generator.py`)
- Scans input files for PyTorch usage
- Categorizes files based on content and path
- Generates prioritized conversion tasks
- Supports task tracking and management

### Rule-based Conversion (`convert_folder.py` and `convert_single_file.py`)
- Implements the core conversion logic using libcst
- Maintains comprehensive API mapping dictionaries
- Handles syntax transformation and import management
- Provides post-processing capabilities

### Sophisticated Conversion `CONVERT.md`
- Accept a json task file as input
- Categorize pytorch script to different categories, and refer to different examples
- Validation and Error Hanlding
- Logging the detailed converison summary in `logs/`


## Conversion Rules

The converter follows comprehensive rules defined in `CLAUDE.md`:

<details>
<summary>details about CLAUDE.md</summary>

### Core Principles
- **Minimal Modification**: Preserve variable names and code structure
- **Device Removal**: Eliminate CUDA-specific device handling
- **Framework Naming**: Replace 'torch' references with 'mindspore' equivalents
- **API Mapping**: Convert PyTorch APIs to MindSpore equivalents

### Key Transformations

| PyTorch | MindSpore |
|---------|-----------|
| `torch.nn.Module.forward()` | `mindspore.nn.Cell.construct()` |
| `torch.expand()` | `mindspore.mint.broadcast_to()` |
| `torch.unflatten()` | `mindspore.ops.reshape()` |
| `torch.Tensor.detach()` | `mindspore.Tensor.clone()` |
| `torch.no_grad()` | `mindspore._no_grad()` |

### Device-Related Code
- Removes `.to(device)`, `device=None`, `.device` calls
- Eliminates CUDA device logic
- Adapts to MindSpore's device context handling

### Parameter Initialization
Converts PyTorch's in-place initializers:
```python
# PyTorch
nn.init.constant_(tensor, val)

# MindSpore
from mindspore.common.initializer import Constant, initializer
tensor.set_data(initializer(Constant(val), tensor.shape, tensor.dtype))
```

## Advanced Features

### Gradient Checkpointing
Handles gradient checkpointing differences between frameworks:
- Removes PyTorch gradient checkpointing logic
- Raises `NotImplementedError` for unsupported MindSpore features
- Preserves non-checkpointing code paths

### Tokenizer Integration
Ensures proper tokenizer output handling:
```python
# Recommended pattern
input_ids = tokenizer("text", return_tensors="np").input_ids
outputs = model(input_ids=Tensor(input_ids))
```

## Testing and Validation

### Generated Test Scripts
The tool generates test scripts that:
- Compare outputs between PyTorch and MindSpore models
- Validate numerical consistency
- Test edge cases and different input scenarios


</details>


## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure MindSpore is properly installed
2. **Device Errors**: Check that device-related code has been properly removed
3. **API Mismatches**: Verify API mappings in `CLAUDE.md`

### Manual Review Required

The tool may require human review for:
- Complex custom operations
- Unsupported PyTorch features
- Domain-specific modifications


## Support

For issues and questions:
1. Check the troubleshooting section
2. Review `CLAUDE.md` for conversion rules
3. Examine examples in the `examples/` directory
4. Open an issue on GitHub

---

**Note**: This tool provides automated conversion capabilities but may require manual review and adjustments for complex models. Always validate the converted models thoroughly before production use. 