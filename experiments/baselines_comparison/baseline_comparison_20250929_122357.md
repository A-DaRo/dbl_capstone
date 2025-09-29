2025-09-29 12:23:57,668 - __main__ - INFO - [_setup_logging:122] - Logging configured with file: /workspace/dbl_capstone/experiments/baselines_comparison/baseline_comparison_20250929_122357.md
2025-09-29 12:23:57,668 - __main__ - INFO - [_setup_logging:123] - Log level set to: DEBUG
2025-09-29 12:23:57,668 - __main__ - INFO - [_setup_logging:124] - Python version: 3.12.11 | packaged by conda-forge | (main, Jun  4 2025, 14:45:31) [GCC 13.3.0]
2025-09-29 12:23:57,668 - __main__ - INFO - [_setup_logging:125] - PyTorch version: 2.8.0+cu129
2025-09-29 12:23:57,668 - __main__ - INFO - [_setup_logging:126] - Current working directory: /workspace/dbl_capstone/experiments/baselines_comparison
2025-09-29 12:23:57,669 - __main__ - INFO - [_setup_logging:127] - Script directory: /workspace/dbl_capstone/experiments/baselines_comparison
2025-09-29 12:23:57,670 - __main__ - INFO - [_setup_logging:137] - Platform: Linux-6.8.0-83-generic-x86_64-with-glibc2.39
2025-09-29 12:23:57,671 - __main__ - INFO - [_setup_logging:138] - Architecture: ('64bit', '')
2025-09-29 12:23:57,697 - __main__ - INFO - [_setup_logging:140] - CUDA available: True, devices: 1
2025-09-29 12:23:57,699 - __main__ - INFO - [_setup_logging:142] -   GPU 0: NVIDIA GeForce RTX 5090
2025-09-29 12:23:57,699 - __main__ - DEBUG - [_validate_configs:180] - Starting configuration validation
2025-09-29 12:23:57,699 - __main__ - DEBUG - [_validate_configs:182] - Checking baseline config at path: /workspace/dbl_capstone/configs/baseline_comparisons/baseline_config.yaml
2025-09-29 12:23:57,699 - __main__ - DEBUG - [_validate_configs:189] - ✓ baseline config file exists
2025-09-29 12:23:57,699 - __main__ - DEBUG - [_validate_configs:182] - Checking mtl config at path: /workspace/dbl_capstone/configs/baseline_comparisons/mtl_config.yaml
2025-09-29 12:23:57,699 - __main__ - DEBUG - [_validate_configs:189] - ✓ mtl config file exists
2025-09-29 12:23:57,699 - __main__ - INFO - [_validate_configs:190] - All configuration files validated successfully
2025-09-29 12:23:57,699 - __main__ - INFO - [run_comparison:390] - 🔬 Starting Comprehensive Baseline Comparison
2025-09-29 12:23:57,699 - __main__ - INFO - [run_comparison:391] - Modes to run: baseline
2025-09-29 12:23:57,699 - __main__ - INFO - [run_comparison:392] - Skip training: True
2025-09-29 12:23:57,699 - __main__ - DEBUG - [run_comparison:393] - Config directory: /workspace/dbl_capstone/configs/baseline_comparisons
2025-09-29 12:23:57,699 - __main__ - DEBUG - [run_comparison:394] - Available config files: ['baseline', 'mtl']
2025-09-29 12:23:57,699 - __main__ - DEBUG - [run_comparison:404] - Checkpoint paths provided: {}
2025-09-29 12:23:57,699 - __main__ - INFO - [run_comparison:408] - 
🚀 Running experiment 1/1: baseline
2025-09-29 12:23:57,699 - __main__ - DEBUG - [run_comparison:417] - Config path for baseline: /workspace/dbl_capstone/configs/baseline_comparisons/baseline_config.yaml
2025-09-29 12:23:57,699 - __main__ - DEBUG - [run_comparison:418] - Checkpoint path for baseline: None
2025-09-29 12:23:57,699 - __main__ - INFO - [run_single_experiment:211] - 
============================================================
2025-09-29 12:23:57,699 - __main__ - INFO - [run_single_experiment:212] - Starting BASELINE Experiment
2025-09-29 12:23:57,699 - __main__ - INFO - [run_single_experiment:213] - Config: /workspace/dbl_capstone/configs/baseline_comparisons/baseline_config.yaml
2025-09-29 12:23:57,699 - __main__ - INFO - [run_single_experiment:214] - Skip training: True
2025-09-29 12:23:57,699 - __main__ - INFO - [run_single_experiment:215] - Checkpoint path: Auto-detect best
2025-09-29 12:23:57,699 - __main__ - INFO - [run_single_experiment:216] - ============================================================
2025-09-29 12:23:57,699 - __main__ - DEBUG - [run_single_experiment:220] - Initializing ExperimentFactory with config: /workspace/dbl_capstone/configs/baseline_comparisons/baseline_config.yaml
2025-09-29 12:23:57,706 - __main__ - DEBUG - [run_single_experiment:222] - ExperimentFactory initialized successfully
2025-09-29 12:23:57,706 - __main__ - DEBUG - [run_single_experiment:237] - Logging experiment configuration details...
2025-09-29 12:23:57,706 - __main__ - INFO - [run_single_experiment:245] - Model type: SegFormerBaseline
2025-09-29 12:23:57,706 - __main__ - INFO - [run_single_experiment:246] - Primary tasks: []
2025-09-29 12:23:57,706 - __main__ - INFO - [run_single_experiment:247] - Auxiliary tasks: []
2025-09-29 12:23:57,706 - __main__ - INFO - [run_single_experiment:248] - Device policy: cuda
2025-09-29 12:23:57,706 - __main__ - INFO - [run_single_experiment:249] - Batch size per GPU: 8
2025-09-29 12:23:57,706 - __main__ - INFO - [run_single_experiment:250] - Learning rate: 6e-05
2025-09-29 12:23:57,706 - __main__ - INFO - [run_single_experiment:251] - Epochs: 50
2025-09-29 12:23:57,706 - __main__ - INFO - [run_single_experiment:252] - Output directory: /workspace/dbl_capstone/experiments/baseline_comparisons/coral_baseline_b2_run
2025-09-29 12:23:57,706 - __main__ - INFO - [run_single_experiment:279] - ⏭️  Skipping training for baseline (evaluation only)
2025-09-29 12:23:57,706 - __main__ - INFO - [run_single_experiment:283] - 🔍 Phase 2: Final Testing & Evaluation for baseline
2025-09-29 12:23:57,706 - __main__ - DEBUG - [run_single_experiment:284] - Starting evaluation phase...
2025-09-29 12:23:57,706 - __main__ - DEBUG - [run_single_experiment:287] - Evaluation started at: 2025-09-29 12:23:57.706763
2025-09-29 12:23:57,706 - __main__ - DEBUG - [run_single_experiment:288] - Using checkpoint: Auto-detect best
2025-09-29 12:24:03,185 - __main__ - ERROR - [run_single_experiment:305] - ❌ Evaluation failed for baseline: CUDA out of memory. Tried to allocate 576.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 370.69 MiB is free. Process 16335 has 25.34 GiB memory in use. Including non-PyTorch memory, this process has 5.55 GiB memory in use. Of the allocated memory 4.89 GiB is allocated by PyTorch, and 77.08 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
Traceback (most recent call last):
  File "/workspace/dbl_capstone/experiments/baselines_comparison/train_val_test_script.py", line 290, in run_single_experiment
    test_metrics = factory.run_evaluation(checkpoint_path=checkpoint_path)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/dbl_capstone/experiments/baselines_comparison/../../src/coral_mtl/ExperimentFactory.py", line 706, in run_evaluation
    final_metrics = evaluator.evaluate()
                    ^^^^^^^^^^^^^^^^^^^^
  File "/workspace/dbl_capstone/experiments/baselines_comparison/../../src/coral_mtl/engine/evaluator.py", line 135, in evaluate
    single_predictions = inferrer.predict(single_image)
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/dbl_capstone/experiments/baselines_comparison/../../src/coral_mtl/engine/inference.py", line 63, in predict
    return self._execute_inference(batch_images_tensor)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/dbl_capstone/experiments/baselines_comparison/../../src/coral_mtl/engine/inference.py", line 82, in _execute_inference
    model_output = self.model(batch_patches)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/dbl_capstone/experiments/baselines_comparison/../../src/coral_mtl/model/core.py", line 70, in forward
    fused_features = self.decoder(features) # Shape: (B, C_decoder, H/4, W/4)
                     ^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/dbl_capstone/experiments/baselines_comparison/../../src/coral_mtl/model/decoders.py", line 95, in forward
    fused_features = self.linear_fuse(fused_features)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/dbl_capstone/experiments/baselines_comparison/../../src/coral_mtl/model/decoders.py", line 16, in forward
    return self.act(self.norm(self.proj(x)))
                              ^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/nn/modules/conv.py", line 548, in forward
    return self._conv_forward(input, self.weight, self.bias)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/main/lib/python3.12/site-packages/torch/nn/modules/conv.py", line 543, in _conv_forward
    return F.conv2d(
           ^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 576.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 370.69 MiB is free. Process 16335 has 25.34 GiB memory in use. Including non-PyTorch memory, this process has 5.55 GiB memory in use. Of the allocated memory 4.89 GiB is allocated by PyTorch, and 77.08 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
2025-09-29 12:24:03,670 - __main__ - ERROR - [run_comparison:431] - ❌ Experiment baseline failed: ❌ Evaluation failed for baseline: CUDA out of memory. Tried to allocate 576.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 370.69 MiB is free. Process 16335 has 25.34 GiB memory in use. Including non-PyTorch memory, this process has 5.55 GiB memory in use. Of the allocated memory 4.89 GiB is allocated by PyTorch, and 77.08 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
2025-09-29 12:24:03,670 - __main__ - INFO - [run_comparison:441] - Total comparison duration: 0:00:05.970593
2025-09-29 12:24:03,670 - __main__ - INFO - [_generate_comparison_summary:453] - 
============================================================
2025-09-29 12:24:03,670 - __main__ - INFO - [_generate_comparison_summary:454] - 📈 BASELINE COMPARISON SUMMARY
2025-09-29 12:24:03,670 - __main__ - INFO - [_generate_comparison_summary:455] - ============================================================
2025-09-29 12:24:03,670 - __main__ - WARNING - [_generate_comparison_summary:500] - ❌ Failed experiments:
2025-09-29 12:24:03,670 - __main__ - WARNING - [_generate_comparison_summary:503] -    - BASELINE: ❌ Evaluation failed for baseline: CUDA out of memory. Tried to allocate 576.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 370.69 MiB is free. Process 16335 has 25.34 GiB memory in use. Including non-PyTorch memory, this process has 5.55 GiB memory in use. Of the allocated memory 4.89 GiB is allocated by PyTorch, and 77.08 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
2025-09-29 12:24:03,670 - __main__ - INFO - [_generate_comparison_summary:505] - ============================================================
2025-09-29 12:24:03,670 - __main__ - INFO - [_save_results:539] - 💾 Full results saved to: baseline_comparison_results_20250929_122403.json
2025-09-29 12:24:03,670 - __main__ - INFO - [_save_results:540] - 💾 Summary results saved to: baseline_comparison_summary_20250929_122403.json
