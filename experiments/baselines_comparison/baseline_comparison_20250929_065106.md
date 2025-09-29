2025-09-29 06:51:06,679 - __main__ - INFO - [_setup_logging:122] - Logging configured with file: /workspace/dbl_capstone/experiments/baselines_comparison/baseline_comparison_20250929_065106.md
2025-09-29 06:51:06,679 - __main__ - INFO - [_setup_logging:123] - Log level set to: DEBUG
2025-09-29 06:51:06,679 - __main__ - INFO - [_setup_logging:124] - Python version: 3.12.11 | packaged by conda-forge | (main, Jun  4 2025, 14:45:31) [GCC 13.3.0]
2025-09-29 06:51:06,679 - __main__ - INFO - [_setup_logging:125] - PyTorch version: 2.8.0+cu129
2025-09-29 06:51:06,679 - __main__ - INFO - [_setup_logging:126] - Current working directory: /workspace/dbl_capstone/experiments/baselines_comparison
2025-09-29 06:51:06,679 - __main__ - INFO - [_setup_logging:127] - Script directory: /workspace/dbl_capstone/experiments/baselines_comparison
2025-09-29 06:51:06,681 - __main__ - INFO - [_setup_logging:137] - Platform: Linux-6.8.0-83-generic-x86_64-with-glibc2.39
2025-09-29 06:51:06,683 - __main__ - INFO - [_setup_logging:138] - Architecture: ('64bit', '')
2025-09-29 06:51:06,829 - __main__ - INFO - [_setup_logging:140] - CUDA available: True, devices: 1
2025-09-29 06:51:06,832 - __main__ - INFO - [_setup_logging:142] -   GPU 0: NVIDIA GeForce RTX 5090
2025-09-29 06:51:06,832 - __main__ - DEBUG - [_validate_configs:180] - Starting configuration validation
2025-09-29 06:51:06,832 - __main__ - DEBUG - [_validate_configs:182] - Checking baseline config at path: /workspace/dbl_capstone/configs/baseline_comparisons/baseline_config.yaml
2025-09-29 06:51:06,832 - __main__ - DEBUG - [_validate_configs:189] - ✓ baseline config file exists
2025-09-29 06:51:06,832 - __main__ - DEBUG - [_validate_configs:182] - Checking mtl config at path: /workspace/dbl_capstone/configs/baseline_comparisons/mtl_config.yaml
2025-09-29 06:51:06,832 - __main__ - DEBUG - [_validate_configs:189] - ✓ mtl config file exists
2025-09-29 06:51:06,832 - __main__ - INFO - [_validate_configs:190] - All configuration files validated successfully
2025-09-29 06:51:06,832 - __main__ - INFO - [run_comparison:390] - 🔬 Starting Comprehensive Baseline Comparison
2025-09-29 06:51:06,832 - __main__ - INFO - [run_comparison:391] - Modes to run: mtl, baseline
2025-09-29 06:51:06,832 - __main__ - INFO - [run_comparison:392] - Skip training: False
2025-09-29 06:51:06,832 - __main__ - DEBUG - [run_comparison:393] - Config directory: /workspace/dbl_capstone/configs/baseline_comparisons
2025-09-29 06:51:06,832 - __main__ - DEBUG - [run_comparison:394] - Available config files: ['baseline', 'mtl']
2025-09-29 06:51:06,832 - __main__ - DEBUG - [run_comparison:404] - Checkpoint paths provided: {}
2025-09-29 06:51:06,832 - __main__ - INFO - [run_comparison:408] - 
🚀 Running experiment 1/2: mtl
2025-09-29 06:51:06,832 - __main__ - DEBUG - [run_comparison:417] - Config path for mtl: /workspace/dbl_capstone/configs/baseline_comparisons/mtl_config.yaml
2025-09-29 06:51:06,832 - __main__ - DEBUG - [run_comparison:418] - Checkpoint path for mtl: None
2025-09-29 06:51:06,832 - __main__ - INFO - [run_single_experiment:211] - 
============================================================
2025-09-29 06:51:06,832 - __main__ - INFO - [run_single_experiment:212] - Starting MTL Experiment
2025-09-29 06:51:06,832 - __main__ - INFO - [run_single_experiment:213] - Config: /workspace/dbl_capstone/configs/baseline_comparisons/mtl_config.yaml
2025-09-29 06:51:06,832 - __main__ - INFO - [run_single_experiment:214] - Skip training: False
2025-09-29 06:51:06,832 - __main__ - INFO - [run_single_experiment:215] - Checkpoint path: Auto-detect best
2025-09-29 06:51:06,833 - __main__ - INFO - [run_single_experiment:216] - ============================================================
2025-09-29 06:51:06,833 - __main__ - DEBUG - [run_single_experiment:220] - Initializing ExperimentFactory with config: /workspace/dbl_capstone/configs/baseline_comparisons/mtl_config.yaml
2025-09-29 06:51:06,839 - __main__ - DEBUG - [run_single_experiment:222] - ExperimentFactory initialized successfully
2025-09-29 06:51:06,839 - __main__ - DEBUG - [run_single_experiment:237] - Logging experiment configuration details...
2025-09-29 06:51:06,839 - __main__ - INFO - [run_single_experiment:245] - Model type: CoralMTL
2025-09-29 06:51:06,839 - __main__ - INFO - [run_single_experiment:246] - Primary tasks: ['genus', 'health']
2025-09-29 06:51:06,839 - __main__ - INFO - [run_single_experiment:247] - Auxiliary tasks: ['fish', 'human_artifacts', 'substrate', 'background', 'biota']
2025-09-29 06:51:06,839 - __main__ - INFO - [run_single_experiment:248] - Device policy: cuda
2025-09-29 06:51:06,839 - __main__ - INFO - [run_single_experiment:249] - Batch size per GPU: 4
2025-09-29 06:51:06,839 - __main__ - INFO - [run_single_experiment:250] - Learning rate: 6e-05
2025-09-29 06:51:06,840 - __main__ - INFO - [run_single_experiment:251] - Epochs: 50
2025-09-29 06:51:06,840 - __main__ - INFO - [run_single_experiment:252] - Output directory: /workspace/dbl_capstone/experiments/baseline_comparisons/coral_mtl_b2_run
2025-09-29 06:51:06,840 - __main__ - INFO - [run_single_experiment:258] - 🎯 Phase 1: Training & Validation for mtl
2025-09-29 06:51:06,840 - __main__ - DEBUG - [run_single_experiment:259] - Starting training phase...
2025-09-29 06:51:06,840 - __main__ - DEBUG - [run_single_experiment:262] - Training started at: 2025-09-29 06:51:06.840057
2025-09-29 06:51:07,454 - coral_mtl.engine.gradient_strategies - INFO - [__init__:313] - Initialized NashMTLStrategy with solver: CCP (cvxpy)
2025-09-29 06:51:15,255 - __main__ - ERROR - [run_single_experiment:275] - ❌ Training failed for mtl: Got unsupported ScalarType BFloat16
Traceback (most recent call last):
  File "/workspace/dbl_capstone/experiments/baselines_comparison/train_val_test_script.py", line 264, in run_single_experiment
    factory.run_training()
  File "/workspace/dbl_capstone/experiments/baselines_comparison/../../src/coral_mtl/ExperimentFactory.py", line 606, in run_training
    trainer.train()
  File "/workspace/dbl_capstone/experiments/baselines_comparison/../../src/coral_mtl/engine/trainer.py", line 473, in train
    train_loss_report = self._train_one_epoch(epoch + 1)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/dbl_capstone/experiments/baselines_comparison/../../src/coral_mtl/engine/trainer.py", line 155, in _train_one_epoch
    update_vec = self.loss_fn.weighting_strategy.compute_update_vector(per_task_grads)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/dbl_capstone/experiments/baselines_comparison/../../src/coral_mtl/engine/gradient_strategies.py", line 420, in compute_update_vector
    weights = self._solve_nash_ccp(G)
              ^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/dbl_capstone/experiments/baselines_comparison/../../src/coral_mtl/engine/gradient_strategies.py", line 353, in _solve_nash_ccp
    GTG = (G @ G.t()).detach().cpu().numpy()
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: Got unsupported ScalarType BFloat16
2025-09-29 06:51:15,359 - __main__ - ERROR - [run_comparison:431] - ❌ Experiment mtl failed: ❌ Training failed for mtl: Got unsupported ScalarType BFloat16
2025-09-29 06:51:15,359 - __main__ - INFO - [run_comparison:408] - 
🚀 Running experiment 2/2: baseline
2025-09-29 06:51:15,359 - __main__ - DEBUG - [run_comparison:417] - Config path for baseline: /workspace/dbl_capstone/configs/baseline_comparisons/baseline_config.yaml
2025-09-29 06:51:15,360 - __main__ - DEBUG - [run_comparison:418] - Checkpoint path for baseline: None
2025-09-29 06:51:15,360 - __main__ - INFO - [run_single_experiment:211] - 
============================================================
2025-09-29 06:51:15,360 - __main__ - INFO - [run_single_experiment:212] - Starting BASELINE Experiment
2025-09-29 06:51:15,360 - __main__ - INFO - [run_single_experiment:213] - Config: /workspace/dbl_capstone/configs/baseline_comparisons/baseline_config.yaml
2025-09-29 06:51:15,360 - __main__ - INFO - [run_single_experiment:214] - Skip training: False
2025-09-29 06:51:15,360 - __main__ - INFO - [run_single_experiment:215] - Checkpoint path: Auto-detect best
2025-09-29 06:51:15,360 - __main__ - INFO - [run_single_experiment:216] - ============================================================
2025-09-29 06:51:15,360 - __main__ - DEBUG - [run_single_experiment:220] - Initializing ExperimentFactory with config: /workspace/dbl_capstone/configs/baseline_comparisons/baseline_config.yaml
2025-09-29 06:51:15,367 - __main__ - DEBUG - [run_single_experiment:222] - ExperimentFactory initialized successfully
2025-09-29 06:51:15,367 - __main__ - DEBUG - [run_single_experiment:237] - Logging experiment configuration details...
2025-09-29 06:51:15,367 - __main__ - INFO - [run_single_experiment:245] - Model type: SegFormerBaseline
2025-09-29 06:51:15,367 - __main__ - INFO - [run_single_experiment:246] - Primary tasks: []
2025-09-29 06:51:15,367 - __main__ - INFO - [run_single_experiment:247] - Auxiliary tasks: []
2025-09-29 06:51:15,367 - __main__ - INFO - [run_single_experiment:248] - Device policy: cuda
2025-09-29 06:51:15,367 - __main__ - INFO - [run_single_experiment:249] - Batch size per GPU: 8
2025-09-29 06:51:15,367 - __main__ - INFO - [run_single_experiment:250] - Learning rate: 6e-05
2025-09-29 06:51:15,367 - __main__ - INFO - [run_single_experiment:251] - Epochs: 50
2025-09-29 06:51:15,367 - __main__ - INFO - [run_single_experiment:252] - Output directory: /workspace/dbl_capstone/experiments/baseline_comparisons/coral_baseline_b2_run
2025-09-29 06:51:15,367 - __main__ - INFO - [run_single_experiment:258] - 🎯 Phase 1: Training & Validation for baseline
2025-09-29 06:51:15,367 - __main__ - DEBUG - [run_single_experiment:259] - Starting training phase...
2025-09-29 06:51:15,367 - __main__ - DEBUG - [run_single_experiment:262] - Training started at: 2025-09-29 06:51:15.367681
2025-09-29 06:51:50,590 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:51:50,590 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:51:50,590 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:51:50,590 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:51:50,590 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:51:50,590 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:51:50,590 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:51:50,590 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:51:50,590 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:51:50,590 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:51:50,590 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:51:50,590 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:51:50,590 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:51:50,590 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:51:50,590 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:51:50,590 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:51:50,590 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:51:50,590 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:51:50,590 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:51:50,590 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:51:50,590 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:51:50,615 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:51:50,700 - __main__ - ERROR - [main:660] - Experiment interrupted by user (Ctrl-C)
Traceback (most recent call last):
  File "/workspace/dbl_capstone/experiments/baselines_comparison/train_val_test_script.py", line 640, in main
    results = comparison.run_comparison(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/dbl_capstone/experiments/baselines_comparison/train_val_test_script.py", line 420, in run_comparison
    experiment_result = self.run_single_experiment(
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/dbl_capstone/experiments/baselines_comparison/train_val_test_script.py", line 264, in run_single_experiment
    factory.run_training()
  File "/workspace/dbl_capstone/experiments/baselines_comparison/../../src/coral_mtl/ExperimentFactory.py", line 606, in run_training
    trainer.train()
  File "/workspace/dbl_capstone/experiments/baselines_comparison/../../src/coral_mtl/engine/trainer.py", line 473, in train
    train_loss_report = self._train_one_epoch(epoch + 1)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/dbl_capstone/experiments/baselines_comparison/../../src/coral_mtl/engine/trainer.py", line 217, in _train_one_epoch
    total_loss.backward()
  File "/venv/main/lib/python3.12/site-packages/torch/_tensor.py", line 647, in backward
    torch.autograd.backward(
  File "/venv/main/lib/python3.12/site-packages/torch/autograd/__init__.py", line 354, in backward
    _engine_run_backward(
  File "/venv/main/lib/python3.12/site-packages/torch/autograd/graph.py", line 829, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/dbl_capstone/experiments/baselines_comparison/train_val_test_script.py", line 162, in _sigint_handler
    raise KeyboardInterrupt()
KeyboardInterrupt
