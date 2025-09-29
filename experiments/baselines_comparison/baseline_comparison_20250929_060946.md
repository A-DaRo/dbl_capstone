2025-09-29 06:09:46,967 - __main__ - INFO - [_setup_logging:122] - Logging configured with file: /workspace/dbl_capstone/experiments/baselines_comparison/baseline_comparison_20250929_060946.md
2025-09-29 06:09:46,967 - __main__ - INFO - [_setup_logging:123] - Log level set to: DEBUG
2025-09-29 06:09:46,967 - __main__ - INFO - [_setup_logging:124] - Python version: 3.12.11 | packaged by conda-forge | (main, Jun  4 2025, 14:45:31) [GCC 13.3.0]
2025-09-29 06:09:46,967 - __main__ - INFO - [_setup_logging:125] - PyTorch version: 2.8.0+cu129
2025-09-29 06:09:46,967 - __main__ - INFO - [_setup_logging:126] - Current working directory: /workspace/dbl_capstone/experiments/baselines_comparison
2025-09-29 06:09:46,967 - __main__ - INFO - [_setup_logging:127] - Script directory: /workspace/dbl_capstone/experiments/baselines_comparison
2025-09-29 06:09:46,969 - __main__ - INFO - [_setup_logging:137] - Platform: Linux-6.8.0-83-generic-x86_64-with-glibc2.39
2025-09-29 06:09:46,970 - __main__ - INFO - [_setup_logging:138] - Architecture: ('64bit', '')
2025-09-29 06:09:47,010 - __main__ - INFO - [_setup_logging:140] - CUDA available: True, devices: 1
2025-09-29 06:09:47,013 - __main__ - INFO - [_setup_logging:142] -   GPU 0: NVIDIA GeForce RTX 5090
2025-09-29 06:09:47,013 - __main__ - DEBUG - [_validate_configs:180] - Starting configuration validation
2025-09-29 06:09:47,013 - __main__ - DEBUG - [_validate_configs:182] - Checking baseline config at path: /workspace/dbl_capstone/configs/baseline_comparisons/baseline_config.yaml
2025-09-29 06:09:47,013 - __main__ - DEBUG - [_validate_configs:189] - ✓ baseline config file exists
2025-09-29 06:09:47,013 - __main__ - DEBUG - [_validate_configs:182] - Checking mtl config at path: /workspace/dbl_capstone/configs/baseline_comparisons/mtl_config.yaml
2025-09-29 06:09:47,013 - __main__ - DEBUG - [_validate_configs:189] - ✓ mtl config file exists
2025-09-29 06:09:47,013 - __main__ - INFO - [_validate_configs:190] - All configuration files validated successfully
2025-09-29 06:09:47,013 - __main__ - INFO - [run_comparison:390] - 🔬 Starting Comprehensive Baseline Comparison
2025-09-29 06:09:47,013 - __main__ - INFO - [run_comparison:391] - Modes to run: mtl, baseline
2025-09-29 06:09:47,013 - __main__ - INFO - [run_comparison:392] - Skip training: False
2025-09-29 06:09:47,013 - __main__ - DEBUG - [run_comparison:393] - Config directory: /workspace/dbl_capstone/configs/baseline_comparisons
2025-09-29 06:09:47,013 - __main__ - DEBUG - [run_comparison:394] - Available config files: ['baseline', 'mtl']
2025-09-29 06:09:47,013 - __main__ - DEBUG - [run_comparison:404] - Checkpoint paths provided: {}
2025-09-29 06:09:47,013 - __main__ - INFO - [run_comparison:408] - 
🚀 Running experiment 1/2: mtl
2025-09-29 06:09:47,013 - __main__ - DEBUG - [run_comparison:417] - Config path for mtl: /workspace/dbl_capstone/configs/baseline_comparisons/mtl_config.yaml
2025-09-29 06:09:47,013 - __main__ - DEBUG - [run_comparison:418] - Checkpoint path for mtl: None
2025-09-29 06:09:47,013 - __main__ - INFO - [run_single_experiment:211] - 
============================================================
2025-09-29 06:09:47,013 - __main__ - INFO - [run_single_experiment:212] - Starting MTL Experiment
2025-09-29 06:09:47,013 - __main__ - INFO - [run_single_experiment:213] - Config: /workspace/dbl_capstone/configs/baseline_comparisons/mtl_config.yaml
2025-09-29 06:09:47,013 - __main__ - INFO - [run_single_experiment:214] - Skip training: False
2025-09-29 06:09:47,013 - __main__ - INFO - [run_single_experiment:215] - Checkpoint path: Auto-detect best
2025-09-29 06:09:47,013 - __main__ - INFO - [run_single_experiment:216] - ============================================================
2025-09-29 06:09:47,013 - __main__ - DEBUG - [run_single_experiment:220] - Initializing ExperimentFactory with config: /workspace/dbl_capstone/configs/baseline_comparisons/mtl_config.yaml
2025-09-29 06:09:47,021 - __main__ - DEBUG - [run_single_experiment:222] - ExperimentFactory initialized successfully
2025-09-29 06:09:47,021 - __main__ - DEBUG - [run_single_experiment:237] - Logging experiment configuration details...
2025-09-29 06:09:47,021 - __main__ - INFO - [run_single_experiment:245] - Model type: CoralMTL
2025-09-29 06:09:47,021 - __main__ - INFO - [run_single_experiment:246] - Primary tasks: ['genus', 'health']
2025-09-29 06:09:47,021 - __main__ - INFO - [run_single_experiment:247] - Auxiliary tasks: ['fish', 'human_artifacts', 'substrate', 'background', 'biota']
2025-09-29 06:09:47,021 - __main__ - INFO - [run_single_experiment:248] - Device policy: cuda
2025-09-29 06:09:47,021 - __main__ - INFO - [run_single_experiment:249] - Batch size per GPU: 8
2025-09-29 06:09:47,021 - __main__ - INFO - [run_single_experiment:250] - Learning rate: 6e-05
2025-09-29 06:09:47,021 - __main__ - INFO - [run_single_experiment:251] - Epochs: 50
2025-09-29 06:09:47,021 - __main__ - INFO - [run_single_experiment:252] - Output directory: /workspace/dbl_capstone/experiments/baseline_comparisons/coral_mtl_b2_run
2025-09-29 06:09:47,021 - __main__ - INFO - [run_single_experiment:258] - 🎯 Phase 1: Training & Validation for mtl
2025-09-29 06:09:47,021 - __main__ - DEBUG - [run_single_experiment:259] - Starting training phase...
2025-09-29 06:09:47,021 - __main__ - DEBUG - [run_single_experiment:262] - Training started at: 2025-09-29 06:09:47.021282
2025-09-29 06:49:07,556 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:49:07,556 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:49:07,556 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:49:07,557 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:49:07,556 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:49:07,556 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:49:07,556 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:49:07,556 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:49:07,556 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:49:07,557 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:49:07,556 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:49:07,556 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:49:07,556 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:49:07,556 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:49:07,556 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:49:07,556 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:49:07,556 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:49:07,557 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:49:07,556 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:49:07,556 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:49:07,556 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:49:07,557 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:49:07,557 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:49:07,557 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:49:07,556 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:49:07,556 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:49:07,556 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:49:07,558 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:49:07,558 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:49:07,556 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:49:07,556 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:49:07,556 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:49:07,556 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:49:07,556 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:49:07,556 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:49:07,556 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:49:07,556 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:49:07,557 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:49:07,659 - __main__ - ERROR - [main:660] - Experiment interrupted by user (Ctrl-C)
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
  File "/workspace/dbl_capstone/experiments/baselines_comparison/../../src/coral_mtl/engine/trainer.py", line 134, in _train_one_epoch
    predictions = self.model(images)
                  ^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/dbl_capstone/experiments/baselines_comparison/../../src/coral_mtl/model/core.py", line 146, in forward
    features: List[torch.Tensor] = self.encoder(images)
                                   ^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/dbl_capstone/experiments/baselines_comparison/../../src/coral_mtl/model/encoder.py", line 90, in forward
    features = self.encoder(x)
               ^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/segmentation_models_pytorch/encoders/mix_transformer.py", line 477, in forward
    return [x, dummy] + self.forward_features(x)[: self._depth - 1]
                        ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/segmentation_models_pytorch/encoders/mix_transformer.py", line 414, in forward_features
    x = blk(x, H, W)
        ^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/segmentation_models_pytorch/encoders/mix_transformer.py", line 164, in forward
    x = x + self.drop_path(self.attn(self.norm1(x), H, W))
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/segmentation_models_pytorch/encoders/mix_transformer.py", line 104, in forward
    attn = attn.softmax(dim=-1)
           ^^^^^^^^^^^^^^^^^^^^
  File "/workspace/dbl_capstone/experiments/baselines_comparison/train_val_test_script.py", line 162, in _sigint_handler
    raise KeyboardInterrupt()
KeyboardInterrupt
