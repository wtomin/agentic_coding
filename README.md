# Agentic Coding: PyTorch to MindSpore Converter

An automated code conversion tool that transforms PyTorch models and configurations to MindSpore, especially designed for `transformers`.

# Quick Integration with mindone

To use this project for converting and testing models in the [mindone](https://github.com/mindspore-lab/mindone.git) repository, follow these steps:

## Clone mindone
```bash
git clone https://github.com/mindspore-lab/mindone.git
cd mindone/
```

## Copy Conversion Assets
Copy the following files and folder from this repository into the `mindone/` folder:
- `plans/` (the entire folder)
- `CLAUDE.md`
- `auto_convert.py`

Afterwards, the folder structure will be:
```bash
mindone/
├── examples/     # originally from mindone
├── mindone/      # originally from mindone
├── scripts/      # originally from mindone
├── ...
├── plans/
│   ├── phase1_modeling_convert.md
│   └── phase2_test_script.md
├── CLAUDE.md
└── auto_convert.py
```


## Prepare Model Source Code

Decide the target model name (e.g., `bert`).
Place the PyTorch model code and configuration files in:
```
mindone/mindone/transformers/models/{model-name}/
```
There should be at least:
- `configuration_{model-name}.py`
- `modeling_{model-name}.py`
- `__init__.py`

## Run Rule-based Conversion

Run the following commands to perform rule-based conversion and update the model folder:
```bash
python auto_convert.py --src_root mindone/mindone/transformers/models/{model-name}/ --dst_root mindone/mindone/transformers/models/{model-name}_ms/
mv mindone/mindone/transformers/models/{model-name}_ms/ mindone/mindone/transformers/models/{model-name}/
```

## Edit the Plans for Your Model
- Open `plans/phase1_modeling_convert.md` and replace all occurrences of `{model-name}` with your target model name (e.g., `bert`).
- Open `plans/phase2_test_script.md` and do the same replacement.

## Launch the Coding Agent
Launch your coding agent (e.g., Claude Code). Claude Code will automatically load `CLAUDE.md` as the system prompt. 

If you are using other coding agent, please set the system prompt accordingly.

## Run the Conversion and Testing Steps
Instruct your coding agent as follows:

### a. Convert the Model

```bash
╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ >   Convert the modeling script following `plans/phase1_modeling_convert.md`.                                        │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

### b. Write and Run the Test Script

```bash
╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ >   Write the test script following `plans/phase2_test_scrip.md`.                                                    │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

---

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

## Documentation

- **`CLAUDE.md`**: Comprehensive conversion rules and technical guidelines
- **`plans/`**: Step-by-step plans for model conversion and testing

## Troubleshooting

1. **Import Errors**: Ensure MindSpore is properly installed
2. **Device Errors**: Check that device-related code has been properly removed
3. **API Mismatches**: Verify API mappings in `CLAUDE.md`

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review `CLAUDE.md` for conversion rules
3. Open an issue on GitHub

---

**Note**: This tool provides automated conversion capabilities but may require manual review and adjustments for complex models. Always validate the converted models thoroughly before production use. 