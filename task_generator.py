#!/usr/bin/env python3
"""
Task Generator for PyTorch to MindSpore Conversion
Scans input files and generates conversion task definitions
"""

import os
import json
import re
import ast
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import shutil
class PyTorchDetector:
    """Detects PyTorch usage in Python files"""
    
    # Target libraries to detect
    TORCH_LIBRARIES = {
        'torch', 'torchvision', 'torchaudio', 'transformers', 'diffusers',
        'timm', 'pytorch_lightning', 'accelerate', 'torchmetrics', 'vllm', 'sglang'
    }
    
    # Patterns for detecting torch usage
    TORCH_PATTERNS = [
        r'\btorch\.',           # torch.tensor, torch.nn, etc.
        r'\bfrom\s+torch\b',    # from torch import
        r'\bimport\s+torch\b',  # import torch
        r'\bTensor\b',          # Tensor class usage
        r'\.cuda\(\)',          # CUDA calls
        r'\.to\(device\)',      # device transfers
        r'torch\.nn\.',         # torch.nn usage
    ]
    
    def __init__(self):
        self.compiled_patterns = [re.compile(pattern) for pattern in self.TORCH_PATTERNS]
    
    def detect_torch_usage(self, file_path: str) -> Tuple[bool, List[str]]:
        """
        Detect if file contains PyTorch code
        Returns: (has_torch, detected_patterns)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            detected = []
            for pattern in self.compiled_patterns:
                if pattern.search(content):
                    detected.append(pattern.pattern)
            
            return len(detected) > 0, detected
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return False, []
    
    def detect_import_usage(self, file_path: str) -> List[str]:
        """Detect specific torch library imports using AST"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if any(lib in alias.name for lib in self.TORCH_LIBRARIES):
                            imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module and any(lib in node.module for lib in self.TORCH_LIBRARIES):
                        imports.append(node.module)
            
            return list(set(imports))
            
        except Exception as e:
            print(f"Error parsing AST for {file_path}: {e}")
            return []


class TaskCategorizer:
    """Categorizes files based on their content and path"""
    
    CATEGORY_KEYWORDS = {
        'modeling': ['model', 'transformer', 'layer', 'attention', 'embedding', 'encoder', 'decoder'],
        'dataset': ['dataset', 'dataloader', 'data', 'preprocess', 'transform'],
        'training': ['train', 'optimizer', 'loss', 'checkpoint', 'trainer'],
        'inference': ['inference', 'predict', 'generate', 'pipeline', 'cli', 'main'],
        'others': []  # fallback category
    }
    
    PRIORITY_MAP = {
        'modeling': 1,
        'inference': 2, 
        'dataset': 3,
        'training': 4,
        'others': 5
    }
    
    def categorize_file(self, file_path: str) -> str:
        """Categorize file based on path and content keywords"""
        file_path_lower = file_path.lower()
        file_name = os.path.basename(file_path_lower)
        
        # Check each category for keyword matches
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            if category != 'others':
                for keyword in keywords:
                    if keyword in file_path_lower or keyword in file_name:
                        return category
        
        return 'others'
    
    def get_priority(self, category: str) -> int:
        """Get priority number for category"""
        return self.PRIORITY_MAP.get(category, 5)


class TaskGenerator:
    """Main task generation orchestrator"""
    
    def __init__(self, inputs_dir: str, output_dir: str = "tasks", logs_dir: str = "logs"):
        self.inputs_dir = Path(inputs_dir)
        self.output_dir = Path(output_dir)
        self.logs_dir = Path(logs_dir)
        self.detector = PyTorchDetector()
        self.categorizer = TaskCategorizer()
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
    
    def scan_files(self) -> List[str]:
        """Scan inputs directory for Python files"""
        python_files = []
        
        for root, dirs, files in os.walk(self.inputs_dir):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        return python_files
    
    def create_task_definition(self, file_path: str, task_id: int) -> Optional[Dict]:
        """Create task definition for a single file"""
        # Detect PyTorch usage
        has_torch, patterns = self.detector.detect_torch_usage(file_path)
        
        if not has_torch:
            return None
        
        # Get imports for additional context
        imports = self.detector.detect_import_usage(file_path)
        
        # Categorize file
        category = self.categorizer.categorize_file(file_path)
        priority = self.categorizer.get_priority(category)
        
        # Generate relative paths with proper separators
        rel_input_path = os.path.relpath(file_path, self.inputs_dir.parent)
        # Normalize path separators for cross-platform compatibility
        rel_input_path = rel_input_path.replace(os.sep, '/')
        output_file = rel_input_path.replace('inputs/', 'outputs/')
        
        # Determine example category
        example_path = f"examples/{category}/" if category != 'others' else "no_example"
        
        task_def = {
            "task_id": f"task_{task_id:03d}",
            "input_file": rel_input_path,
            "output_file": output_file,
            "category": category,
            "example": example_path,
            "log_file": f"logs/task_{task_id:03d}.log",
            "priority": priority,
            "detected_patterns": patterns,
            "torch_imports": imports,
            "created_at": datetime.now().isoformat(),
            "status": "pending"
        }
        
        return task_def
    
    def generate_tasks(self) -> List[Dict]:
        """Generate all task definitions"""
        python_files = self.scan_files()
        tasks = []
        task_id = 1
        
        print(f"Scanning {len(python_files)} Python files...")
        
        for file_path in python_files:
            task_def = self.create_task_definition(file_path, task_id)
            if task_def:
                tasks.append(task_def)
                print(f"[+] Created task for {file_path} (category: {task_def['category']})")
                task_id += 1
            else:
                # If no PyTorch usage detected, simply copy the file to the output directory
                file_path = file_path.replace(os.sep, '/')
                output_file = file_path.replace('inputs/', 'outputs/')
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                shutil.copy(file_path, output_file)
                print(f"[-] Skipped {file_path} (no PyTorch usage detected)")
        
        # Sort by priority then by task_id
        tasks.sort(key=lambda x: (x['priority'], x['task_id']))
        
        return tasks
    
    def save_task_files(self, tasks: List[Dict]) -> None:
        """Save individual task JSON files"""
        for task in tasks:
            filename = f"{task['task_id']}_p{task['priority']}.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(task, f, indent=2, ensure_ascii=False)
    
    def save_task_summary(self, tasks: List[Dict]) -> None:
        """Save summary of all tasks"""
        summary = {
            "total_tasks": len(tasks),
            "categories": {},
            "priority_distribution": {},
            "generated_at": datetime.now().isoformat(),
            "tasks": tasks
        }
        
        # Calculate category and priority distributions
        for task in tasks:
            category = task['category']
            priority = task['priority']
            
            summary['categories'][category] = summary['categories'].get(category, 0) + 1
            summary['priority_distribution'][str(priority)] = summary['priority_distribution'].get(str(priority), 0) + 1
        
        # Save summary
        with open(self.output_dir / 'task_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n=== Task Generation Summary ===")
        print(f"Total tasks generated: {summary['total_tasks']}")
        print(f"Categories: {dict(summary['categories'])}")
        print(f"Priority distribution: {dict(summary['priority_distribution'])}")
    
    def run(self) -> List[Dict]:
        """Run the complete task generation process"""
        print(f"Starting task generation for directory: {self.inputs_dir}")
        
        tasks = self.generate_tasks()
        
        if tasks:
            self.save_task_files(tasks)
            self.save_task_summary(tasks)
            print(f"\n[+] Generated {len(tasks)} conversion tasks")
            print(f"[+] Task files saved to: {self.output_dir}")
        else:
            print("No PyTorch files found for conversion")
        
        return tasks


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate PyTorch to MindSpore conversion tasks")
    parser.add_argument("--inputs", default="inputs", help="Input directory to scan")
    parser.add_argument("--output", default="tasks", help="Output directory for task files")
    parser.add_argument("--logs", default="logs", help="Logs directory")
    
    args = parser.parse_args()
    
    generator = TaskGenerator(args.inputs, args.output, args.logs)
    tasks = generator.run()
    
    return tasks


if __name__ == "__main__":
    main()