# Agentic Coding: PyTorch to MindSpore Converter

An automated code conversion tool that transforms PyTorch models and configurations to MindSpore, especially designed for `transformers`.

## Overview

This project provides a sophisticated automated conversion system that:

- **Converts PyTorch models to MindSpore**: Automatically transforms PyTorch model architectures, preserving functionality and precision
- **Handles configuration files**: Processes PyTorch configuration files 
- **Maintains code structure**: Uses the minimal modification principle to preserve variable names and overall code structure
- **Generates test scripts**: Creates comprehensive test scripts to validate conversion accuracy

## Installation

### Prerequisites
- Python 3.7 or higher
- PyTorch (for testing and validation)
- MindSpore (target framework)
- Coding Agent (such as Claude or Cursor)

### Dependencies
Install the required dependencies:

```bash
pip install libcst # For rule-based auto-conversion
pip install torch  # For validation purposes
pip install mindspore  # Target framework
```

### Clone the Repository
```bash
git clone https://github.com/your-username/agentic_coding.git
cd agentic_coding
```
## Quick Start

```bash
mv example_inputs/ inputs/
python auto_convert.py --src_root ./inputs --dst_root ./outputs  # partially convert torch scripts to mindspore
```

Then, start the conversion with your coding agent like this:
```
Follow the instruction in @INITIAL_PLAN.md and start the code conversion task.
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

### Step1: Rule-based Partial Conversion

Convert PyTorch models to MindSpore partially using the automated converter:

```bash
python auto_convert.py --src_root ./inputs --dst_root ./outputs
```

**Parameters**:
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

### Step2: Conversion with Coding Agent

If you have partially converted files in the `outputs/` directory, the code agent will:
- Detect existing MindSpore files
- Apply additional conversion rules
- Complete remaining transformations
- Generate diff reports for human review

<details>

<summary> The Role of INITIAL_PLAN.md: Guiding the Code Agent </summary>

The INITIAL_PLAN.md file is the central blueprint that directs the code agent's entire conversion process. It is not just a set of instructions, but a detailed operational plan. When you instruct the agent to follow this plan, it initiates a structured workflow to ensure an accurate and high-quality migration.

Here is how the agent interprets and executes the plan:

Analysis and Scoping: The agent begins by scanning the inputs and outputs directories as defined in the plan. This initial analysis allows it to scope the task:

It identifies the target PyTorch files that require conversion.
It detects if any MindSpore files already exist in outputs. If so, it recognizes this as a partially completed task and switches to an iterative, diff-based editing mode, ensuring it builds upon existing work rather than starting from scratch.
Reference-Based Generation: The agent studies the examples section to understand the desired end state. The provided cohere2 example acts as a "Rosetta Stone," teaching the agent the target coding style, the expected file structure, and the correct format for configuration and testing scripts.

Systematic Conversion and Reflection: Following the Workflow, the agent converts each file. After converting a script, it enters a crucial reflection step. It reviews its own work against a set of global principles (like the Minimal Modification Principle, API mapping rules, etc.) defined in the plan. This self-correction phase ensures that subtle, framework-specific details are not missed.

Validation and Testing: Finally, the agent uses its understanding from the Reference section to generate a new test script for the converted model. This final step validates that the generated MindSpore model is functionally equivalent to the original PyTorch model, completing the development cycle.
</details>

<details>
   
<summary> Customizing INITIAL_PLAN.md for Your Task </summary>

You can—and should—modify INITIAL_PLAN.md to fit the specific needs of your conversion project. A well-crafted plan leads to a more accurate and efficient conversion. Here’s how to tailor each section:
Inputs Section:

Purpose: To tell the agent which source files to convert.
How to Modify: Place your source PyTorch model scripts and any relevant configuration files into the inputs/ directory. The agent will automatically detect and process them.
Outputs Section:

Purpose: To define the desired result of the conversion.
How to Modify: You can specify the naming conventions for the output files here. For instance, you might instruct the agent that all converted MindSpore files should have a specific suffix or be placed in a particular subdirectory.
Reference Section:

Purpose: To provide a complete, high-quality example of a correctly converted model.
How to Modify: This is the most effective way to guide the agent. If your target model has a unique structure or requires a specific testing methodology, create a small, representative example and place it in the examples/ folder. The agent will learn and replicate the patterns from your example, significantly improving the quality of the final output.
Workflow Section:

Purpose: To define the exact sequence of operations for the agent.
How to Modify: For most standard conversions, the default workflow is sufficient. However, for complex projects (e.g., those with multiple interdependent models or custom validation steps), you can modify this section. For example, you could add a step to first convert a base model class before converting several child models that inherit from it, ensuring dependencies are handled correctly.
</details>

If you are using Claude Code, simply input the following text in the terminal to trigger this entire process base on the current `INITIAL_PLAN.md`:

```bash
╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ >   Follow the instruction in @INITIAL_PLAN.md and start the code conversion task.                                   │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```


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

## Documentation

- **`CLAUDE.md`**: Comprehensive conversion rules and technical guidelines
- **`INITIAL_PLAN.md`**: Project methodology and workflow
- **Examples**: Reference implementations in `examples/` directory



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
