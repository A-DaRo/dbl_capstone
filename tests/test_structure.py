"""
Basic structural tests for coral_mtl modules.
Tests that can run without external dependencies.
"""
import unittest
import os
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestModuleStructure(unittest.TestCase):
    """Test basic module structure and organization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.src_path = Path(__file__).parent.parent / "src" / "coral_mtl"
        
    def test_src_directory_exists(self):
        """Test that source directory structure exists."""
        self.assertTrue(self.src_path.exists())
        self.assertTrue(self.src_path.is_dir())
        
    def test_core_modules_exist(self):
        """Test that expected core modules exist."""
        expected_modules = [
            "__init__.py",
            "ExperimentFactory.py",
            "data",
            "engine", 
            "metrics",
            "model",
            "scripts",
            "utils"
        ]
        
        for module_name in expected_modules:
            module_path = self.src_path / module_name
            self.assertTrue(
                module_path.exists(), 
                f"Expected module {module_name} not found at {module_path}"
            )
    
    def test_submodule_structure(self):
        """Test that submodules have expected structure."""
        submodule_files = {
            "data": ["__init__.py", "dataset.py", "augmentations.py"],
            "engine": ["__init__.py", "trainer.py", "evaluator.py", "losses.py", "optimizer.py", "inference.py"],
            "metrics": ["__init__.py", "metrics.py", "metrics_storer.py"],
            "model": ["__init__.py", "core.py", "encoder.py", "decoders.py", "attention.py"],
            "utils": ["__init__.py", "task_splitter.py", "visualization.py"],
            "scripts": ["__init__.py"]
        }
        
        for submodule, expected_files in submodule_files.items():
            submodule_path = self.src_path / submodule
            self.assertTrue(submodule_path.exists(), f"Submodule {submodule} missing")
            
            for expected_file in expected_files:
                file_path = submodule_path / expected_file
                self.assertTrue(
                    file_path.exists(),
                    f"Expected file {expected_file} missing in {submodule}"
                )
    
    def test_config_structure(self):
        """Test configuration directory structure."""
        config_path = Path(__file__).parent.parent / "configs"
        self.assertTrue(config_path.exists())
        
        expected_configs = [
            "task_definitions.yaml",
            "baseline_segformer.yaml",
            "three_tier_baseline_config.yaml"
        ]
        
        for config_file in expected_configs:
            config_file_path = config_path / config_file
            self.assertTrue(
                config_file_path.exists(),
                f"Expected config file {config_file} missing"
            )
    
    def test_test_directory_structure(self):
        """Test that test directory has proper structure."""
        test_path = Path(__file__).parent
        
        # Check basic test files exist
        expected_test_files = [
            "__init__.py",
            "conftest.py"
        ]
        
        for test_file in expected_test_files:
            test_file_path = test_path / test_file
            self.assertTrue(
                test_file_path.exists(),
                f"Expected test file {test_file} missing"
            )


class TestConfigurationFiles(unittest.TestCase):
    """Test configuration file structure and content."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config_path = Path(__file__).parent.parent / "configs"
    
    def test_task_definitions_yaml_structure(self):
        """Test task definitions YAML has expected structure."""
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML not available")
            
        task_def_path = self.config_path / "task_definitions.yaml"
        self.assertTrue(task_def_path.exists())
        
        with open(task_def_path, 'r') as f:
            task_defs = yaml.safe_load(f)
        
        # Verify it's a dictionary
        self.assertIsInstance(task_defs, dict)
        
        # Verify it has expected top-level tasks
        expected_tasks = ['genus', 'health', 'morphology', 'bleaching']
        for task in expected_tasks:
            if task in task_defs:  # Not all may be present
                task_def = task_defs[task]
                self.assertIn('id2label', task_def)
                self.assertIsInstance(task_def['id2label'], dict)
    
    def test_baseline_config_structure(self):
        """Test baseline configuration structure."""
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML not available")
            
        baseline_config_path = self.config_path / "baseline_segformer.yaml"
        if not baseline_config_path.exists():
            self.skipTest("Baseline config file not found")
            
        with open(baseline_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Verify basic structure
        expected_sections = ['model', 'data', 'trainer']
        for section in expected_sections:
            if section in config:  # May not all be present
                self.assertIsInstance(config[section], dict)


class TestProjectSpecifications(unittest.TestCase):
    """Test project specification files exist and are readable."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.spec_path = Path(__file__).parent.parent / "project_specification"
    
    def test_specification_files_exist(self):
        """Test that specification files exist."""
        if not self.spec_path.exists():
            self.skipTest("Project specification directory not found")
            
        expected_specs = [
            "tests_specification.md",
            "technical_specification.md"
        ]
        
        for spec_file in expected_specs:
            spec_file_path = self.spec_path / spec_file
            if spec_file_path.exists():
                # Verify file is readable and non-empty
                with open(spec_file_path, 'r') as f:
                    content = f.read()
                    self.assertGreater(len(content), 100, f"{spec_file} seems too short")


class TestImportStructure(unittest.TestCase):
    """Test what can be imported without external dependencies."""
    
    def test_basic_python_imports(self):
        """Test basic Python imports work."""
        try:
            import os
            import sys
            import pathlib
            import unittest
            self.assertTrue(True)  # If we get here, imports work
        except ImportError as e:
            self.fail(f"Basic Python imports failed: {e}")
    
    def test_optional_imports(self):
        """Test optional imports and graceful handling."""
        optional_modules = ['yaml', 'numpy', 'torch']
        
        for module_name in optional_modules:
            try:
                __import__(module_name)
                # Module available - good!
            except ImportError:
                # Module not available - that's expected in some environments
                pass
    
    def test_coral_mtl_init_file(self):
        """Test that coral_mtl __init__.py exists and is importable."""
        init_path = Path(__file__).parent.parent / "src" / "coral_mtl" / "__init__.py"
        self.assertTrue(init_path.exists())
        
        # Try to read the file (not import due to dependencies)
        try:
            with open(init_path, 'r') as f:
                content = f.read()
                # File should exist and be readable
                self.assertIsInstance(content, str)
        except Exception as e:
            self.fail(f"Could not read coral_mtl __init__.py: {e}")


if __name__ == '__main__':
    unittest.main()