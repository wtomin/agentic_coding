# Execute Code Conversion Project

Given a task file, run following workflow:
1. identify the input file python, and run the python conversion script:
```bash
python conver_single_file.py --input_file /path/to/input/py --output_file /path/to/output/py
```

2. identity the category, and read the examples of corresponding category. If it is `others` category, skip reading examples.
3. run further conversion on `output_file`, because some torch-related codes need manual conversion.
4. validate the converted code, such as syntax validation, import resolution check, basic MindSpore API compatibility check and cross-reference with conversion rules.
5. summerize the edits and save it to `log_file`.

## Task Explanations

### TASK FILE

An example of task file: `tasks/task_{file_id}_p{priority}.json`

```json
{
  "task_id": "task_001",
  "input_file": "path/to/input/file.py",
  "category": "modeling|dataset|training|inference|others", 
  "example": "examples/modeling/|examples/dataset/|examples/training/|examples/inference/|no_example",
  "log_file": "logs/task_001.log",
  "output_file": "path/to/output/file.py",
  "created_at": "timestamp",
  "priority": 1
}
```

### Examples as Conversion Reference
Get conversion examples for each category:
- `examples/modeling/`: Examples, rules, and prompts for modeling script conversion
- `examples/dataset/`: Examples, rules, and prompts for dataset script conversion  
- `examples/training/`: Examples, rules, and prompts for training script conversion
- `examples/inference/`: Examples, rules, and prompts for inference script conversion

### Priority

The conversion priority of the four categories:
- `modeling/`: first priority
- `dataset/`: third priority 
- `training/`: fourth priority
- `inference/`: second priority
- `others`: fifth priority

## Conversion Process

### Task Processor Design
Each conversion task runs independently with this workflow, for example:

```
=== TASK METADATA ===
Task ID: task_001
Input File: src/models/transformer.py
Category: modeling
Started: 2024-01-01 10:00:00
Completed: 2024-01-01 10:05:30

=== CONVERSION SUMMARY ===
Lines Changed: 45
Imports Converted: 8
Functions Modified: 12
Classes Modified: 3

=== DETAILED CHANGES ===
[Line 15] import torch → import mindspore as ms
[Line 23] torch.nn.Linear → ms.nn.Dense
[Line 67] torch.tensor() → ms.Tensor()
...

=== ISSUES ENCOUNTERED ===
- Custom CUDA kernel at line 234 (manual review needed)
- Unsupported operation: torch.jit.script (line 156)

=== VALIDATION ===
Syntax Check: PASSED
Import Check: PASSED
Basic Functionality: PASSED

=== LLM INTERACTION LOG ===
[Timestamp] System prompt loaded
[Timestamp] Examples context loaded  
[Timestamp] Conversion started
[Timestamp] First pass completed
[Timestamp] Validation performed
[Timestamp] Final output generated
```

Save the above log into `log_file` defined in the task file, for example, `"log_file": "logs/task_001.log",`.

The converted file will be saved to `output_file` defined in the task file. 

### Code validation and Error handling
- Failed tasks are retried up to 3 times
- Partial conversions are saved for manual review
- Dependency conflicts are logged for batch resolution

For each converted file:
- Syntax validation with Python AST
- Import resolution check
- Basic MindSpore API compatibility check
- Cross-reference with conversion rules


