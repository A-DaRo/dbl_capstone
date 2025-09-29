2025-09-29 06:02:35,919 - __main__ - INFO - [_setup_logging:122] - Logging configured with file: /workspace/dbl_capstone/experiments/baselines_comparison/baseline_comparison_20250929_060235.md
2025-09-29 06:02:35,919 - __main__ - INFO - [_setup_logging:123] - Log level set to: DEBUG
2025-09-29 06:02:35,919 - __main__ - INFO - [_setup_logging:124] - Python version: 3.12.11 | packaged by conda-forge | (main, Jun  4 2025, 14:45:31) [GCC 13.3.0]
2025-09-29 06:02:35,919 - __main__ - INFO - [_setup_logging:125] - PyTorch version: 2.8.0+cu129
2025-09-29 06:02:35,919 - __main__ - INFO - [_setup_logging:126] - Current working directory: /workspace/dbl_capstone/experiments/baselines_comparison
2025-09-29 06:02:35,919 - __main__ - INFO - [_setup_logging:127] - Script directory: /workspace/dbl_capstone/experiments/baselines_comparison
2025-09-29 06:02:35,920 - __main__ - INFO - [_setup_logging:137] - Platform: Linux-6.8.0-83-generic-x86_64-with-glibc2.39
2025-09-29 06:02:35,921 - __main__ - INFO - [_setup_logging:138] - Architecture: ('64bit', '')
2025-09-29 06:02:35,964 - __main__ - INFO - [_setup_logging:140] - CUDA available: True, devices: 1
2025-09-29 06:02:35,968 - __main__ - INFO - [_setup_logging:142] -   GPU 0: NVIDIA GeForce RTX 5090
2025-09-29 06:02:35,968 - __main__ - DEBUG - [_validate_configs:180] - Starting configuration validation
2025-09-29 06:02:35,968 - __main__ - DEBUG - [_validate_configs:182] - Checking baseline config at path: /workspace/dbl_capstone/configs/baseline_comparisons/baseline_config.yaml
2025-09-29 06:02:35,968 - __main__ - DEBUG - [_validate_configs:189] - ✓ baseline config file exists
2025-09-29 06:02:35,968 - __main__ - DEBUG - [_validate_configs:182] - Checking mtl config at path: /workspace/dbl_capstone/configs/baseline_comparisons/mtl_config.yaml
2025-09-29 06:02:35,968 - __main__ - DEBUG - [_validate_configs:189] - ✓ mtl config file exists
2025-09-29 06:02:35,968 - __main__ - INFO - [_validate_configs:190] - All configuration files validated successfully
2025-09-29 06:02:35,968 - __main__ - INFO - [run_comparison:390] - 🔬 Starting Comprehensive Baseline Comparison
2025-09-29 06:02:35,968 - __main__ - INFO - [run_comparison:391] - Modes to run: mtl, baseline
2025-09-29 06:02:35,968 - __main__ - INFO - [run_comparison:392] - Skip training: False
2025-09-29 06:02:35,968 - __main__ - DEBUG - [run_comparison:393] - Config directory: /workspace/dbl_capstone/configs/baseline_comparisons
2025-09-29 06:02:35,968 - __main__ - DEBUG - [run_comparison:394] - Available config files: ['baseline', 'mtl']
2025-09-29 06:02:35,968 - __main__ - DEBUG - [run_comparison:404] - Checkpoint paths provided: {}
2025-09-29 06:02:35,968 - __main__ - INFO - [run_comparison:408] - 
🚀 Running experiment 1/2: mtl
2025-09-29 06:02:35,968 - __main__ - DEBUG - [run_comparison:417] - Config path for mtl: /workspace/dbl_capstone/configs/baseline_comparisons/mtl_config.yaml
2025-09-29 06:02:35,968 - __main__ - DEBUG - [run_comparison:418] - Checkpoint path for mtl: None
2025-09-29 06:02:35,968 - __main__ - INFO - [run_single_experiment:211] - 
============================================================
2025-09-29 06:02:35,968 - __main__ - INFO - [run_single_experiment:212] - Starting MTL Experiment
2025-09-29 06:02:35,968 - __main__ - INFO - [run_single_experiment:213] - Config: /workspace/dbl_capstone/configs/baseline_comparisons/mtl_config.yaml
2025-09-29 06:02:35,968 - __main__ - INFO - [run_single_experiment:214] - Skip training: False
2025-09-29 06:02:35,968 - __main__ - INFO - [run_single_experiment:215] - Checkpoint path: Auto-detect best
2025-09-29 06:02:35,968 - __main__ - INFO - [run_single_experiment:216] - ============================================================
2025-09-29 06:02:35,968 - __main__ - DEBUG - [run_single_experiment:220] - Initializing ExperimentFactory with config: /workspace/dbl_capstone/configs/baseline_comparisons/mtl_config.yaml
2025-09-29 06:02:35,975 - __main__ - DEBUG - [run_single_experiment:222] - ExperimentFactory initialized successfully
2025-09-29 06:02:35,975 - __main__ - DEBUG - [run_single_experiment:237] - Logging experiment configuration details...
2025-09-29 06:02:35,975 - __main__ - INFO - [run_single_experiment:245] - Model type: CoralMTL
2025-09-29 06:02:35,975 - __main__ - INFO - [run_single_experiment:246] - Primary tasks: ['genus', 'health']
2025-09-29 06:02:35,975 - __main__ - INFO - [run_single_experiment:247] - Auxiliary tasks: ['fish', 'human_artifacts', 'substrate', 'background', 'biota']
2025-09-29 06:02:35,975 - __main__ - INFO - [run_single_experiment:248] - Device policy: cuda
2025-09-29 06:02:35,975 - __main__ - INFO - [run_single_experiment:249] - Batch size per GPU: 8
2025-09-29 06:02:35,975 - __main__ - INFO - [run_single_experiment:250] - Learning rate: 6e-05
2025-09-29 06:02:35,975 - __main__ - INFO - [run_single_experiment:251] - Epochs: 50
2025-09-29 06:02:35,975 - __main__ - INFO - [run_single_experiment:252] - Output directory: /workspace/dbl_capstone/experiments/baseline_comparisons/coral_mtl_b2_run
2025-09-29 06:02:35,975 - __main__ - INFO - [run_single_experiment:258] - 🎯 Phase 1: Training & Validation for mtl
2025-09-29 06:02:35,975 - __main__ - DEBUG - [run_single_experiment:259] - Starting training phase...
2025-09-29 06:02:35,975 - __main__ - DEBUG - [run_single_experiment:262] - Training started at: 2025-09-29 06:02:35.975658
2025-09-29 06:02:36,590 - coral_mtl.engine.gradient_strategies - INFO - [__init__:189] - Initialized IMGradStrategy with solver: QP (cvxopt)
2025-09-29 06:02:45,460 - __main__ - ERROR - [run_single_experiment:275] - ❌ Training failed for mtl: CUDA out of memory. Tried to allocate 128.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 60.69 MiB is free. Process 11956 has 10.16 GiB memory in use. Including non-PyTorch memory, this process has 20.98 GiB memory in use. Of the allocated memory 20.19 GiB is allocated by PyTorch, and 186.63 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
Traceback (most recent call last):
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
  File "/workspace/dbl_capstone/experiments/baselines_comparison/../../src/coral_mtl/model/core.py", line 147, in forward
    logits_at_quarter_res = self.decoder(features)
                            ^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/dbl_capstone/experiments/baselines_comparison/../../src/coral_mtl/model/decoders.py", line 245, in forward
    f_projected_enrichment = self.attn_proj[task](f_enriched)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/dbl_capstone/experiments/baselines_comparison/../../src/coral_mtl/model/decoders.py", line 16, in forward
    return self.act(self.norm(self.proj(x)))
                    ^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/nn/modules/batchnorm.py", line 193, in forward
    return F.batch_norm(
           ^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/nn/functional.py", line 2817, in batch_norm
    return torch.batch_norm(
           ^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 128.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 60.69 MiB is free. Process 11956 has 10.16 GiB memory in use. Including non-PyTorch memory, this process has 20.98 GiB memory in use. Of the allocated memory 20.19 GiB is allocated by PyTorch, and 186.63 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
2025-09-29 06:02:45,876 - __main__ - ERROR - [run_comparison:431] - ❌ Experiment mtl failed: ❌ Training failed for mtl: CUDA out of memory. Tried to allocate 128.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 60.69 MiB is free. Process 11956 has 10.16 GiB memory in use. Including non-PyTorch memory, this process has 20.98 GiB memory in use. Of the allocated memory 20.19 GiB is allocated by PyTorch, and 186.63 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
2025-09-29 06:02:45,876 - __main__ - INFO - [run_comparison:408] - 
🚀 Running experiment 2/2: baseline
2025-09-29 06:02:45,876 - __main__ - DEBUG - [run_comparison:417] - Config path for baseline: /workspace/dbl_capstone/configs/baseline_comparisons/baseline_config.yaml
2025-09-29 06:02:45,876 - __main__ - DEBUG - [run_comparison:418] - Checkpoint path for baseline: None
2025-09-29 06:02:45,876 - __main__ - INFO - [run_single_experiment:211] - 
============================================================
2025-09-29 06:02:45,876 - __main__ - INFO - [run_single_experiment:212] - Starting BASELINE Experiment
2025-09-29 06:02:45,877 - __main__ - INFO - [run_single_experiment:213] - Config: /workspace/dbl_capstone/configs/baseline_comparisons/baseline_config.yaml
2025-09-29 06:02:45,877 - __main__ - INFO - [run_single_experiment:214] - Skip training: False
2025-09-29 06:02:45,877 - __main__ - INFO - [run_single_experiment:215] - Checkpoint path: Auto-detect best
2025-09-29 06:02:45,877 - __main__ - INFO - [run_single_experiment:216] - ============================================================
2025-09-29 06:02:45,877 - __main__ - DEBUG - [run_single_experiment:220] - Initializing ExperimentFactory with config: /workspace/dbl_capstone/configs/baseline_comparisons/baseline_config.yaml
2025-09-29 06:02:45,884 - __main__ - DEBUG - [run_single_experiment:222] - ExperimentFactory initialized successfully
2025-09-29 06:02:45,884 - __main__ - DEBUG - [run_single_experiment:237] - Logging experiment configuration details...
2025-09-29 06:02:45,884 - __main__ - INFO - [run_single_experiment:245] - Model type: SegFormerBaseline
2025-09-29 06:02:45,884 - __main__ - INFO - [run_single_experiment:246] - Primary tasks: []
2025-09-29 06:02:45,884 - __main__ - INFO - [run_single_experiment:247] - Auxiliary tasks: []
2025-09-29 06:02:45,884 - __main__ - INFO - [run_single_experiment:248] - Device policy: cuda
2025-09-29 06:02:45,884 - __main__ - INFO - [run_single_experiment:249] - Batch size per GPU: 8
2025-09-29 06:02:45,884 - __main__ - INFO - [run_single_experiment:250] - Learning rate: 6e-05
2025-09-29 06:02:45,884 - __main__ - INFO - [run_single_experiment:251] - Epochs: 50
2025-09-29 06:02:45,884 - __main__ - INFO - [run_single_experiment:252] - Output directory: /workspace/dbl_capstone/experiments/baseline_comparisons/coral_baseline_b2_run
2025-09-29 06:02:45,884 - __main__ - INFO - [run_single_experiment:258] - 🎯 Phase 1: Training & Validation for baseline
2025-09-29 06:02:45,884 - __main__ - DEBUG - [run_single_experiment:259] - Starting training phase...
2025-09-29 06:02:45,884 - __main__ - DEBUG - [run_single_experiment:262] - Training started at: 2025-09-29 06:02:45.884411
2025-09-29 06:02:55,438 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:02:55,438 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:02:55,438 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:02:55,438 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:02:55,438 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:02:55,438 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:02:55,438 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:02:55,438 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:02:55,438 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:02:55,438 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:02:55,438 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:02:55,438 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:02:55,438 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:02:55,438 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:02:55,438 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:02:55,438 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:02:55,438 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:02:55,438 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:02:55,438 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:02:55,438 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:02:55,439 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:02:55,438 - __main__ - ERROR - [_sigint_handler:152] - Received KeyboardInterrupt (Ctrl-C). Flushing logs and exiting...
NoneType: None
2025-09-29 06:02:55,549 - __main__ - ERROR - [main:660] - Experiment interrupted by user (Ctrl-C)
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
  File "/workspace/dbl_capstone/experiments/baselines_comparison/../../src/coral_mtl/engine/trainer.py", line 178, in _train_one_epoch
    loss_dict_or_tensor = self.loss_fn(predictions, masks_for_loss)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/dbl_capstone/experiments/baselines_comparison/../../src/coral_mtl/engine/losses.py", line 99, in forward
    if torch.isnan(loss_primary):
       ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/dbl_capstone/experiments/baselines_comparison/train_val_test_script.py", line 162, in _sigint_handler
    raise KeyboardInterrupt()
KeyboardInterrupt
