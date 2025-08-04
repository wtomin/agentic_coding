# Input Processing Strategy

## Optimized Batch Processing Process

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

### PHASE 2: Batch Conversion with Diff Review

```python
conversion_results = {
    'completed': [],
    'failed': [],
    'unchanged': []
}

torch_to_mindspore_files = scan_and_categorize_inputs()

for file_path in torch_to_mindspore_files:
    try:
        # 1. Load original
        original_content = open(f'inputs/{file_path}').read()
        
        # 2. Generate conversion
        converted_content = convert_file(original_content, file_path)
        
        # 3. Create diff for human review
        diff_txt = create_diff_summary(original_content, converted_content)
        
        # 4. Wait for human verification of changes
        print(f"\n{'='*50}")
        print(f"DIFF FOR: {file_path}")
        print(f"{'='*50}")
        print(diff_txt)
        
        # 5. Human verification via file marker
        verification_file = f"outputs/{file_path}.verification_pending"
        Path(verification_file).write_text(f"Review needed for: {file_path}\n{diff_txt}")
        
        # 6. Continue with backup created
        create_backup(f'outputs/{file_path}')
        Path(f"outputs/{file_path}").write_text(converted_content)
        
        conversion_results['completed'].append(file_path)
        
    except Exception as e:
        conversion_results['failed'].append((file_path, str(e)))
```

### PHASE 3: Structure Replication

**GUARANTEE:** Every converted file maintains exact relative path
```
inputs/subfolder/model.py → outputs/subfolder/model.py
inputs/utils/data.py → outputs/utils/data.py
```

## Critical Components

### Required Helper Functions
```python
def create_diff_summary(original: str, converted: str) -> str:
    """Generate concise diff for human review"""
    import difflib
    lines1 = original.splitlines()
    lines2 = converted.splitlines()
    diff = difflib.unified_diff(lines1, lines2, lineterm='', n=3)
    return '\n'.join(diff)

def create_backup(output_path: str):
    """Create .backup file before writing"""
    backup_path = output_path + '.backup'
    if Path(output_path).exists():
        shutil.copy2(output_path, backup_path)

def convert_file(content: str, filepath: str) -> str:
    """Apply PyTorch→MindSpore conversion rules"""
    # 1. Remove device code
    # 2. Transform imports
    # 3. Update API calls per CLAUDE.md
    # 4. Handle naming changes
    return converted_content

```

### Folder Structure Automation
```python
auto_convert.py:
- [ ] auto_scan() - return list[file_paths]
- [ ] convert_batch() - apply to all files
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

## Final Execution Command
```bash
python auto_convert.py inputs/ outputs/

# Results in:
# outputs/ - mirror structure with converted files
# outputs/*.backup - original files
# outputs/*.diff - change summaries
# outputs/_conversion_report.json
```

**Key: Batch process all torch files → generate diffs → human reviews diffs → can modify outputs afterward**