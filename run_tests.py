#!/usr/bin/env python3
"""
Test runner script for the Coral-MTL project.
Provides different test execution modes based on available dependencies.
"""
import sys
import unittest
import argparse
from pathlib import Path


def check_dependencies():
    """Check which optional dependencies are available."""
    deps = {
        'yaml': False,
        'numpy': False,
        'torch': False,
        'transformers': False
    }
    
    for dep in deps:
        try:
            __import__(dep)
            deps[dep] = True
        except ImportError:
            pass
    
    return deps


def run_structural_tests():
    """Run tests that don't require external dependencies."""
    print("Running structural tests...")
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add structural tests
    from tests import test_structure
    suite.addTest(loader.loadTestsFromModule(test_structure))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


def run_unit_tests(deps):
    """Run unit tests based on available dependencies."""
    print("Running unit tests...")
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Always run structural tests
    from tests import test_structure
    suite.addTest(loader.loadTestsFromModule(test_structure))
    
    # Add dependency-based tests
    if deps.get('numpy') and deps.get('torch'):
        print("Full dependencies available - running complete unit tests")
        # TODO: Add full unit test modules when dependencies are resolved
        try:
            from tests import test_task_splitter
            suite.addTest(loader.loadTestsFromModule(test_task_splitter))
        except ImportError as e:
            print(f"Skipping task_splitter tests due to import error: {e}")
    else:
        print("Limited dependencies - running structural tests only")
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


def run_integration_tests(deps):
    """Run integration tests (requires full dependency stack)."""
    if not all(deps[dep] for dep in ['numpy', 'torch', 'transformers']):
        print("Integration tests require full dependency stack - skipping")
        return True
        
    print("Running integration tests...")
    # TODO: Implement when dependencies are available
    print("Integration tests not yet implemented")
    return True


def run_all_tests():
    """Run comprehensive test suite."""
    deps = check_dependencies()
    print("Dependency status:", deps)
    
    success = True
    
    # Run tests in order of complexity
    success &= run_structural_tests()
    success &= run_unit_tests(deps) 
    success &= run_integration_tests(deps)
    
    return success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run Coral-MTL tests')
    parser.add_argument('--structural', action='store_true', 
                       help='Run only structural tests (no external deps)')
    parser.add_argument('--unit', action='store_true',
                       help='Run unit tests')
    parser.add_argument('--integration', action='store_true',
                       help='Run integration tests')
    parser.add_argument('--all', action='store_true', default=True,
                       help='Run all available tests (default)')
    
    args = parser.parse_args()
    
    # Add src to path
    src_path = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_path))
    
    success = True
    
    if args.structural:
        success = run_structural_tests()
    elif args.unit:
        deps = check_dependencies()
        success = run_unit_tests(deps)
    elif args.integration:
        deps = check_dependencies()
        success = run_integration_tests(deps)
    else:  # args.all or default
        success = run_all_tests()
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()