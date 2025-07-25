# Agentic Coding: PyTorch to MindSpore Converter

An automated code conversion tool that transforms PyTorch models and configurations to MindSpore.

## Overview

This project provides a sophisticated automated conversion system that:

- **Converts PyTorch models to MindSpore**: Automatically transforms PyTorch model architectures, preserving functionality and precision
- **Handles configuration files**: Processes PyTorch configuration files 
- **Maintains code structure**: Uses the minimal modification principle to preserve variable names and overall code structure
- **Generates test scripts**: Creates comprehensive test scripts to validate conversion accuracy
- **Leverages AI assistance**: Integrates with Claude Code (or other Coding Agent, such as Cursor) for complex conversion scenarios.

## Installation

### Prerequisites
- Python 3.7 or higher
- PyTorch (for testing and validation)
- MindSpore (target framework)
- Coding Agent (such as Claude or Cursor)

### Dependencies
Install the required dependencies:

```bash
pip install libcst # For relu-based auto-conversion
pip install torch  # For validation purposes
pip install mindspore  # Target framework
```

### Clone the Repository
```bash
git clone https://github.com/your-username/agentic_coding.git
cd agentic_coding
```

## Project Structure

```
agentic_coding/
├── auto_convert.py          # Main conversion script
├── inputs/                  # Input PyTorch files to be converted
├── outputs/                 # Generated MindSpore files
├── examples/                # Reference examples (cohere2 model)
│   ├── configuration_cohere2.py
│   ├── modeling_cohere2_torch.py
│   ├── modeling_cohere2.py
│   └── test_modeling_cohere2.py
├── example_inputs/         # Sample input files for testing
│   ├── configuration_ibert.py
│   ├── modeling_ibert.py
│   └── quant_modules.py
├── CLAUDE.md              # Detailed conversion rules and guidelines
├── INITIAL_PLAN.md        # Project workflow and methodology
└── README.md              # This file
```

## Usage Examples

### Step1: Relu-based Partial Conversion

Convert PyTorch models to MindSpore partially using the automated converter:

```bash
python auto_convert.py --src_root ./inputs --dst_root ./outputs
```

### Parameters
- `--src_root`: Directory containing PyTorch source files (model files and configurations)
- `--dst_root`: Target directory for converted MindSpore files


Detailed steps include:

1. **Place your PyTorch files in the `inputs/` directory:**
   ```
   inputs/
   ├── configuration_mymodel.py
   └── modeling_mymodel.py
   ```

2. **Run the conversion:**
   ```bash
   python auto_convert.py --src_root ./inputs --dst_root ./outputs
   ```

3. **Check the converted files in `outputs/`:**
   ```
   outputs/
   ├── configuration_mymodel.py
   └── modeling_mymodel.py  # Partially converted to MindSpore
   ```

There are other options for rule-based auto-conversion. Please refer to [MSConverter](https://github.com/zhtmike/MSConverter).

You may continue the conversion process with a Coding Agent, which will apply additional conversion rules and complete the remaining transformations.

### Step2: Continue Conversion with Coding Agent

If you have partially converted files in the `outputs/` directory, the code agent will:
- Detect existing MindSpore files
- Apply additional conversion rules
- Complete remaining transformations
- Generate diff reports for human review

If you are using Claude Code, simpky input the following text in the terminal:

```bash
╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ >   Follow the instruction in @INITIAL_PLAN.md and start the code conversion task.                                   │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## Conversion Rules

The converter follows comprehensive rules defined in `CLAUDE.md`:

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


## Examples

Refer to the `examples/` directory for complete conversion examples:

- **cohere2 Model**: Complete example showing PyTorch to MindSpore conversion
- **Configuration Handling**: How configuration files are processed
- **Test Script Generation**: Validation and testing approaches

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

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test thoroughly with example models
5. Submit a pull request

## Documentation

- **`CLAUDE.md`**: Comprehensive conversion rules and technical guidelines
- **`INITIAL_PLAN.md`**: Project methodology and workflow
- **Examples**: Reference implementations in `examples/` directory



## Support

For issues and questions:
1. Check the troubleshooting section
2. Review `CLAUDE.md` for conversion rules
3. Examine examples in the `examples/` directory
4. Open an issue on GitHub

---

**Note**: This tool provides automated conversion capabilities but may require manual review and adjustments for complex models. Always validate the converted models thoroughly before production use. 