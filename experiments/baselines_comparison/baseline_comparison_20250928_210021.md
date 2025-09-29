2025-09-28 21:00:21,154 - __main__ - INFO - [_setup_logging:122] - Logging configured with file: /workspace/dbl_capstone/experiments/baselines_comparison/baseline_comparison_20250928_210021.md
2025-09-28 21:00:21,154 - __main__ - INFO - [_setup_logging:123] - Log level set to: DEBUG
2025-09-28 21:00:21,154 - __main__ - INFO - [_setup_logging:124] - Python version: 3.12.11 | packaged by conda-forge | (main, Jun  4 2025, 14:45:31) [GCC 13.3.0]
2025-09-28 21:00:21,154 - __main__ - INFO - [_setup_logging:125] - PyTorch version: 2.8.0+cu129
2025-09-28 21:00:21,154 - __main__ - INFO - [_setup_logging:126] - Current working directory: /workspace/dbl_capstone/experiments/baselines_comparison
2025-09-28 21:00:21,154 - __main__ - INFO - [_setup_logging:127] - Script directory: /workspace/dbl_capstone/experiments/baselines_comparison
2025-09-28 21:00:21,156 - __main__ - INFO - [_setup_logging:137] - Platform: Linux-6.8.0-83-generic-x86_64-with-glibc2.39
2025-09-28 21:00:21,157 - __main__ - INFO - [_setup_logging:138] - Architecture: ('64bit', '')
2025-09-28 21:00:21,304 - __main__ - INFO - [_setup_logging:140] - CUDA available: True, devices: 1
2025-09-28 21:00:21,307 - __main__ - INFO - [_setup_logging:142] -   GPU 0: NVIDIA GeForce RTX 5090
2025-09-28 21:00:21,307 - __main__ - DEBUG - [_validate_configs:180] - Starting configuration validation
2025-09-28 21:00:21,307 - __main__ - DEBUG - [_validate_configs:182] - Checking baseline config at path: /workspace/dbl_capstone/configs/baseline_comparisons/baseline_config.yaml
2025-09-28 21:00:21,307 - __main__ - DEBUG - [_validate_configs:189] - ✓ baseline config file exists
2025-09-28 21:00:21,307 - __main__ - DEBUG - [_validate_configs:182] - Checking mtl config at path: /workspace/dbl_capstone/configs/baseline_comparisons/mtl_config.yaml
2025-09-28 21:00:21,307 - __main__ - DEBUG - [_validate_configs:189] - ✓ mtl config file exists
2025-09-28 21:00:21,307 - __main__ - INFO - [_validate_configs:190] - All configuration files validated successfully
2025-09-28 21:00:21,307 - __main__ - INFO - [run_comparison:390] - 🔬 Starting Comprehensive Baseline Comparison
2025-09-28 21:00:21,307 - __main__ - INFO - [run_comparison:391] - Modes to run: mtl, baseline
2025-09-28 21:00:21,307 - __main__ - INFO - [run_comparison:392] - Skip training: False
2025-09-28 21:00:21,308 - __main__ - DEBUG - [run_comparison:393] - Config directory: /workspace/dbl_capstone/configs/baseline_comparisons
2025-09-28 21:00:21,308 - __main__ - DEBUG - [run_comparison:394] - Available config files: ['baseline', 'mtl']
2025-09-28 21:00:21,308 - __main__ - DEBUG - [run_comparison:404] - Checkpoint paths provided: {}
2025-09-28 21:00:21,308 - __main__ - INFO - [run_comparison:408] - 
🚀 Running experiment 1/2: mtl
2025-09-28 21:00:21,308 - __main__ - DEBUG - [run_comparison:417] - Config path for mtl: /workspace/dbl_capstone/configs/baseline_comparisons/mtl_config.yaml
2025-09-28 21:00:21,308 - __main__ - DEBUG - [run_comparison:418] - Checkpoint path for mtl: None
2025-09-28 21:00:21,308 - __main__ - INFO - [run_single_experiment:211] - 
============================================================
2025-09-28 21:00:21,308 - __main__ - INFO - [run_single_experiment:212] - Starting MTL Experiment
2025-09-28 21:00:21,308 - __main__ - INFO - [run_single_experiment:213] - Config: /workspace/dbl_capstone/configs/baseline_comparisons/mtl_config.yaml
2025-09-28 21:00:21,308 - __main__ - INFO - [run_single_experiment:214] - Skip training: False
2025-09-28 21:00:21,308 - __main__ - INFO - [run_single_experiment:215] - Checkpoint path: Auto-detect best
2025-09-28 21:00:21,308 - __main__ - INFO - [run_single_experiment:216] - ============================================================
2025-09-28 21:00:21,308 - __main__ - DEBUG - [run_single_experiment:220] - Initializing ExperimentFactory with config: /workspace/dbl_capstone/configs/baseline_comparisons/mtl_config.yaml
2025-09-28 21:00:21,315 - __main__ - DEBUG - [run_single_experiment:222] - ExperimentFactory initialized successfully
2025-09-28 21:00:21,315 - __main__ - DEBUG - [run_single_experiment:237] - Logging experiment configuration details...
2025-09-28 21:00:21,315 - __main__ - INFO - [run_single_experiment:245] - Model type: CoralMTL
2025-09-28 21:00:21,315 - __main__ - INFO - [run_single_experiment:246] - Primary tasks: ['genus', 'health']
2025-09-28 21:00:21,315 - __main__ - INFO - [run_single_experiment:247] - Auxiliary tasks: ['fish', 'human_artifacts', 'substrate', 'background', 'biota']
2025-09-28 21:00:21,315 - __main__ - INFO - [run_single_experiment:248] - Device policy: cuda
2025-09-28 21:00:21,315 - __main__ - INFO - [run_single_experiment:249] - Batch size per GPU: 8
2025-09-28 21:00:21,315 - __main__ - INFO - [run_single_experiment:250] - Learning rate: 6e-05
2025-09-28 21:00:21,315 - __main__ - INFO - [run_single_experiment:251] - Epochs: 50
2025-09-28 21:00:21,315 - __main__ - INFO - [run_single_experiment:252] - Output directory: /workspace/dbl_capstone/experiments/baseline_comparisons/coral_mtl_b2_run
2025-09-28 21:00:21,315 - __main__ - INFO - [run_single_experiment:258] - 🎯 Phase 1: Training & Validation for mtl
2025-09-28 21:00:21,315 - __main__ - DEBUG - [run_single_experiment:259] - Starting training phase...
2025-09-28 21:00:21,315 - __main__ - DEBUG - [run_single_experiment:262] - Training started at: 2025-09-28 21:00:21.315303
2025-09-28 23:32:43,082 - __main__ - INFO - [run_single_experiment:268] - Training duration: 2:32:21.766643
2025-09-28 23:32:43,082 - __main__ - INFO - [run_single_experiment:271] - ✅ Training completed successfully for mtl
2025-09-28 23:32:43,082 - __main__ - INFO - [run_single_experiment:283] - 🔍 Phase 2: Final Testing & Evaluation for mtl
2025-09-28 23:32:43,082 - __main__ - DEBUG - [run_single_experiment:284] - Starting evaluation phase...
2025-09-28 23:32:43,082 - __main__ - DEBUG - [run_single_experiment:287] - Evaluation started at: 2025-09-28 23:32:43.082688
2025-09-28 23:32:43,082 - __main__ - DEBUG - [run_single_experiment:288] - Using checkpoint: Auto-detect best
2025-09-28 23:50:57,328 - __main__ - INFO - [run_single_experiment:294] - Evaluation duration: 0:18:14.245422
2025-09-28 23:50:57,329 - __main__ - INFO - [_log_key_metrics:331] - 📊 Key Metrics for MTL:
2025-09-28 23:50:57,330 - __main__ - DEBUG - [_log_key_metrics:332] - Full metrics structure for mtl: {
  "tasks": {
    "genus": {
      "ungrouped": {
        "task_summary": {
          "mIoU": 0.3988144926552631,
          "mPrecision": 0.5469384216246881,
          "mRecall": 0.5369424366760911,
          "mF1-Score": 0.5238650500701681,
          "pixel_accuracy": 0.9143569566765599
        },
        "per_class": {
          "unlabeled": {
            "IoU": 0.9241583434035467,
            "Precision": 0.9630122237895932,
            "Recall": 0.9581689926754524,
            "F1-Score": 0.9605845034238185,
            "support": "703026221"
          },
          "other coral dead": {
            "IoU": 0.04108020002246522,
            "Precision": 0.23250223898950806,
            "Recall": 0.04752491776284588,
            "F1-Score": 0.07891841574084063,
            "support": "1054268"
          },
          "other coral bleached": {
            "IoU": 0.6829972163689594,
            "Precision": 0.7586522888911507,
            "Recall": 0.8725943360381841,
            "F1-Score": 0.8116439049644008,
            "support": "2434515"
          },
          "other coral alive": {
            "IoU": 0.33965265999551875,
            "Precision": 0.6306513632108687,
            "Recall": 0.423994338393843,
            "F1-Score": 0.5070757072160106,
            "support": "17700984"
          },
          "massive/meandering bleached": {
            "IoU": 0.5319675378144917,
            "Precision": 0.846714405195113,
            "Recall": 0.5886583725222652,
            "F1-Score": 0.6944893082700666,
            "support": "5219586"
          },
          "massive/meandering alive": {
            "IoU": 0.6455463801553926,
            "Precision": 0.7206977659018177,
            "Recall": 0.8609325489838789,
            "F1-Score": 0.7845982197043054,
            "support": "40545023"
          },
          "branching bleached": {
            "IoU": 0.4009727065802742,
            "Precision": 0.8035988333821826,
            "Recall": 0.4445366460710323,
            "F1-Score": 0.5724204400227535,
            "support": "1881143"
          },
          "branching dead": {
            "IoU": 0.36293842682887284,
            "Precision": 0.5534882502113706,
            "Recall": 0.5131981186941578,
            "F1-Score": 0.5325822791177968,
            "support": "8396083"
          },
          "branching alive": {
            "IoU": 0.16021223868787102,
            "Precision": 0.21692751813038236,
            "Recall": 0.3799557084664142,
            "F1-Score": 0.27617746709698776,
            "support": "2044183"
          },
          "massive/meandering dead": {
            "IoU": 0.16564880828134038,
            "Precision": 0.3164807019902735,
            "Recall": 0.25792355749528817,
            "F1-Score": 0.28421735106575857,
            "support": "9399692"
          },
          "acropora alive": {
            "IoU": 0.3606457248537872,
            "Precision": 0.6692010153511423,
            "Recall": 0.4388881317952042,
            "F1-Score": 0.5301096652363959,
            "support": "2649019"
          },
          "table acropora alive": {
            "IoU": 0.2814184454370381,
            "Precision": 0.3513976395383826,
            "Recall": 0.5856006378269125,
            "F1-Score": 0.43922958412083424,
            "support": "1429855"
          },
          "pocillopora alive": {
            "IoU": 0.7537265141484235,
            "Precision": 0.8262687729432027,
            "Recall": 0.8956711269817604,
            "F1-Score": 0.8595713277613516,
            "support": "8169378"
          },
          "table acropora dead": {
            "IoU": 0.011562743989446194,
            "Precision": 0.014759446961056591,
            "Recall": 0.05068052447660757,
            "F1-Score": 0.022861150350090058,
            "support": "677933"
          },
          "meandering bleached": {
            "IoU": 0.4470871404681845,
            "Precision": 0.649296660880119,
            "Recall": 0.5894237173336344,
            "F1-Score": 0.6179132243874902,
            "support": "885850"
          },
          "stylophora alive": {
            "IoU": 0.2642777729086922,
            "Precision": 0.3170133872619606,
            "Recall": 0.6137015371809355,
            "F1-Score": 0.41806915943902884,
            "support": "437294"
          },
          "meandering alive": {
            "IoU": 0.6709683203890687,
            "Precision": 0.7893120026471997,
            "Recall": 0.8173559309517215,
            "F1-Score": 0.8030892174339251,
            "support": "13541250"
          },
          "meandering dead": {
            "IoU": 0.1337996874613621,
            "Precision": 0.1849170739690613,
            "Recall": 0.32615471651950156,
            "F1-Score": 0.23601997591117124,
            "support": "2591307"
          }
        },
        "TIDE_errors": {
          "classification_error": 0.018398185165560955,
          "background_error": 0.03577287708009988,
          "missed_error": 0.03147198107777806
        },
        "BIoU": 0.11457209995613145
      },
      "grouped": {
        "task_summary": {
          "mIoU": 0.48643825570710064,
          "mPrecision": 0.6203186948535933,
          "mRecall": 0.6602646281880057,
          "mF1-Score": 0.6241139360863852,
          "pixel_accuracy": 0.9195085009750044
        },
        "per_class": {
          "unlabeled": {
            "IoU": 0.9241583434035467,
            "Precision": 0.9630122237895932,
            "Recall": 0.9581689926754524,
            "F1-Score": 0.9605845034238185,
            "support": "703026221"
          },
          "other coral": {
            "IoU": 0.38227386379606787,
            "Precision": 0.6694231265076694,
            "Recall": 0.471230854024964,
            "F1-Score": 0.5531087200712147,
            "support": "21189767"
          },
          "massive/meandering": {
            "IoU": 0.5969002320446193,
            "Precision": 0.7190384184972745,
            "Recall": 0.7784672373533746,
            "F1-Score": 0.7475736054974049,
            "support": "55164301"
          },
          "branching": {
            "IoU": 0.3609644022549157,
            "Precision": 0.5286459145693909,
            "Recall": 0.532274190394946,
            "F1-Score": 0.5304538482518005,
            "support": "12321409"
          },
          "acropora": {
            "IoU": 0.3606457248537872,
            "Precision": 0.6692010153511423,
            "Recall": 0.4388881317952042,
            "F1-Score": 0.5301096652363959,
            "support": "2649019"
          },
          "table acropora": {
            "IoU": 0.17779093249333547,
            "Precision": 0.2184962162768028,
            "Recall": 0.48831808512051494,
            "F1-Score": 0.3019057586340206,
            "support": "2107788"
          },
          "pocillopora": {
            "IoU": 0.7537265141484235,
            "Precision": 0.8262687729432027,
            "Recall": 0.8956711269817604,
            "F1-Score": 0.8595713277613516,
            "support": "8169378"
          },
          "meandering": {
            "IoU": 0.5572065154605177,
            "Precision": 0.6717691784853035,
            "Recall": 0.7656614981648987,
            "F1-Score": 0.7156488364624306,
            "support": "17018407"
          },
          "stylophora": {
            "IoU": 0.2642777729086922,
            "Precision": 0.3170133872619606,
            "Recall": 0.6137015371809355,
            "F1-Score": 0.41806915943902884,
            "support": "437294"
          }
        },
        "TIDE_errors": {
          "classification_error": 0.013246640867116485,
          "background_error": 0.03577287708009988,
          "missed_error": 0.03147198107777806
        },
        "BIoU": 0.11812449087697233
      }
    },
    "health": {
      "ungrouped": {
        "task_summary": {
          "mIoU": 0.4065487132236443,
          "mPrecision": 0.5457162972931795,
          "mRecall": 0.5471977627423236,
          "mF1-Score": 0.5314867472963161,
          "pixel_accuracy": 0.9155230315364127
        },
        "per_class": {
          "unlabeled": {
            "IoU": 0.9248986887142818,
            "Precision": 0.9630253693446081,
            "Recall": 0.9589518212294389,
            "F1-Score": 0.9609842784318787,
            "support": "703026221"
          },
          "other coral dead": {
            "IoU": 0.03883489244272628,
            "Precision": 0.1633075595884721,
            "Recall": 0.048481031388603275,
            "F1-Score": 0.07476624577252994,
            "support": "1054268"
          },
          "other coral bleached": {
            "IoU": 0.6805493160560803,
            "Precision": 0.7466850470613438,
            "Recall": 0.8848394854827347,
            "F1-Score": 0.8099129368642345,
            "support": "2434515"
          },
          "other coral alive": {
            "IoU": 0.3472651086688745,
            "Precision": 0.648104017454651,
            "Recall": 0.42795722542882364,
            "F1-Score": 0.5155111736130086,
            "support": "17700984"
          },
          "massive/meandering bleached": {
            "IoU": 0.5774920814398878,
            "Precision": 0.8366271656329679,
            "Recall": 0.6508933850309201,
            "F1-Score": 0.7321647927548013,
            "support": "5219586"
          },
          "massive/meandering alive": {
            "IoU": 0.6457471878031318,
            "Precision": 0.7219936860945086,
            "Recall": 0.8594461766614363,
            "F1-Score": 0.7847465182852588,
            "support": "40545023"
          },
          "branching bleached": {
            "IoU": 0.4272084737402712,
            "Precision": 0.789657007064667,
            "Recall": 0.4820659567082354,
            "F1-Score": 0.5986630287034244,
            "support": "1881143"
          },
          "branching dead": {
            "IoU": 0.38361980253299355,
            "Precision": 0.5858526110830811,
            "Recall": 0.526361876127237,
            "F1-Score": 0.5545162071700629,
            "support": "8396083"
          },
          "branching alive": {
            "IoU": 0.17034122765332774,
            "Precision": 0.22652024668705792,
            "Recall": 0.4071739174036767,
            "F1-Score": 0.291096688091356,
            "support": "2044183"
          },
          "massive/meandering dead": {
            "IoU": 0.16756126070706365,
            "Precision": 0.32113276427130694,
            "Recall": 0.2594713741684302,
            "F1-Score": 0.28702778405921103,
            "support": "9399692"
          },
          "acropora alive": {
            "IoU": 0.3377421029511601,
            "Precision": 0.6872443981301206,
            "Recall": 0.3990816977907671,
            "F1-Score": 0.5049435196904927,
            "support": "2649019"
          },
          "table acropora alive": {
            "IoU": 0.29171480165861746,
            "Precision": 0.3756084166741651,
            "Recall": 0.5663609247091488,
            "F1-Score": 0.4516706029597912,
            "support": "1429855"
          },
          "pocillopora alive": {
            "IoU": 0.7581355277369065,
            "Precision": 0.8300940263526514,
            "Recall": 0.8973901073007027,
            "F1-Score": 0.8624312696903267,
            "support": "8169378"
          },
          "table acropora dead": {
            "IoU": 0.009597897264398551,
            "Precision": 0.012748902166689556,
            "Recall": 0.03738127514075875,
            "F1-Score": 0.019013306763821448,
            "support": "677933"
          },
          "meandering bleached": {
            "IoU": 0.5036286379037831,
            "Precision": 0.631470697286747,
            "Recall": 0.7132742563639443,
            "F1-Score": 0.669884338736584,
            "support": "885850"
          },
          "stylophora alive": {
            "IoU": 0.27922253916401685,
            "Precision": 0.33561371029009346,
            "Recall": 0.6243145343864769,
            "F1-Score": 0.43655037433360305,
            "support": "437294"
          },
          "meandering alive": {
            "IoU": 0.6454032969135397,
            "Precision": 0.7619618890109324,
            "Recall": 0.8083961968060556,
            "F1-Score": 0.784492529125464,
            "support": "13541250"
          },
          "meandering dead": {
            "IoU": 0.12891399467453593,
            "Precision": 0.18524583708316805,
            "Recall": 0.2977184872344342,
            "F1-Score": 0.22838585628784172,
            "support": "2591307"
          }
        },
        "TIDE_errors": {
          "classification_error": 0.017887477972069545,
          "background_error": 0.035103420821987814,
          "missed_error": 0.03148606966952885
        },
        "BIoU": 0.12042858774130918
      },
      "grouped": {
        "task_summary": {
          "mIoU": 0.6087960113717259,
          "mPrecision": 0.7341945437532049,
          "mRecall": 0.7157528221037814,
          "mF1-Score": 0.723928701876609,
          "pixel_accuracy": 0.9254017630401912
        },
        "per_class": {
          "unlabeled": {
            "IoU": 0.9248986887142818,
            "Precision": 0.9630253693446081,
            "Recall": 0.9589518212294389,
            "F1-Score": 0.9609842784318787,
            "support": "703026221"
          },
          "alive coral": {
            "IoU": 0.6640067668944513,
            "Precision": 0.7770418146732841,
            "Recall": 0.8202929191268868,
            "F1-Score": 0.7980818108494743,
            "support": "86516986"
          },
          "bleached coral": {
            "IoU": 0.6161429860333889,
            "Precision": 0.8180834100623456,
            "Recall": 0.7139641001223096,
            "F1-Score": 0.7624857346881553,
            "support": "10421094"
          },
          "dead coral": {
            "IoU": 0.23013560384478174,
            "Precision": 0.3786275809325819,
            "Recall": 0.36980244793649053,
            "F1-Score": 0.37416298353692756,
            "support": "22119283"
          }
        },
        "TIDE_errors": {
          "classification_error": 0.008008746468290981,
          "background_error": 0.035103420821987814,
          "missed_error": 0.03148606966952885
        },
        "BIoU": 0.1322200460455979
      }
    },
    "fish": {
      "ungrouped": {
        "task_summary": {
          "mIoU": 0.7650369900617493,
          "mPrecision": 0.8668028447219019,
          "mRecall": 0.8294203132995214,
          "mF1-Score": 0.8470927648810136,
          "pixel_accuracy": 0.9962813014886805
        },
        "per_class": {
          "unlabeled": {
            "IoU": 0.9962653993177546,
            "Precision": 0.9977974119908697,
            "Recall": 0.9984612213939366,
            "F1-Score": 0.9981292063252102,
            "support": "816782866"
          },
          "fish": {
            "IoU": 0.5338085808057441,
            "Precision": 0.7358082774529342,
            "Recall": 0.6603794052051062,
            "F1-Score": 0.6960563234368169,
            "support": "5300718"
          }
        },
        "TIDE_errors": {
          "classification_error": 0.0,
          "background_error": 0.0015288567056461239,
          "missed_error": 0.0021898418056721563
        },
        "BIoU": 0.19093821799256344
      }
    },
    "human_artifacts": {
      "ungrouped": {
        "task_summary": {
          "mIoU": 0.7273973028539837,
          "mPrecision": 0.8749170149937497,
          "mRecall": 0.7947752139805654,
          "mF1-Score": 0.8320375689791639,
          "pixel_accuracy": 0.9923563271152719
        },
        "per_class": {
          "unlabeled": {
            "IoU": 0.992764667672694,
            "Precision": 0.9947919250468319,
            "Recall": 0.9979514821494077,
            "F1-Score": 0.9963691988097341,
            "support": "802808235"
          },
          "human": {
            "IoU": 0.6998269436139382,
            "Precision": 0.887670679283797,
            "Recall": 0.7678245990254533,
            "F1-Score": 0.8234096373670398,
            "support": "14910111"
          },
          "transect tools": {
            "IoU": 0.5376223822751542,
            "Precision": 0.7539393588791395,
            "Recall": 0.6520286299888643,
            "F1-Score": 0.6992905260388541,
            "support": "1239260"
          },
          "transect line": {
            "IoU": 0.6793752178541484,
            "Precision": 0.8632660967652301,
            "Recall": 0.7612961447585364,
            "F1-Score": 0.8090809137010277,
            "support": "3125978"
          }
        },
        "TIDE_errors": {
          "classification_error": 0.0005410715025298443,
          "background_error": 0.002000486364169994,
          "missed_error": 0.005102115018027163
        },
        "BIoU": 0.37182436331301455
      }
    },
    "substrate": {
      "ungrouped": {
        "task_summary": {
          "mIoU": 0.555353888133594,
          "mPrecision": 0.7353601763190145,
          "mRecall": 0.6802651448180504,
          "mF1-Score": 0.7042194144189277,
          "pixel_accuracy": 0.7970959313061763
        },
        "per_class": {
          "unlabeled": {
            "IoU": 0.7681699150046479,
            "Precision": 0.8542029933887497,
            "Recall": 0.8840846549804606,
            "F1-Score": 0.8688869870321583,
            "support": "523448470"
          },
          "seagrass": {
            "IoU": 0.47630140425884804,
            "Precision": 0.7404013877630311,
            "Recall": 0.5717905420003145,
            "F1-Score": 0.6452630917843866,
            "support": "2892895"
          },
          "sand": {
            "IoU": 0.7191698798110272,
            "Precision": 0.8357832655134482,
            "Recall": 0.8375141969926929,
            "F1-Score": 0.8366478359777674,
            "support": "72597241"
          },
          "algae covered substrate": {
            "IoU": 0.4638832460224314,
            "Precision": 0.724746587379196,
            "Recall": 0.5630877250130822,
            "F1-Score": 0.6337708246649655,
            "support": "68937214"
          },
          "unknown hard substrate": {
            "IoU": 0.3904813133418862,
            "Precision": 0.5628652168922318,
            "Recall": 0.5604382962494482,
            "F1-Score": 0.5616491348645346,
            "support": "108065538"
          },
          "rubble": {
            "IoU": 0.5141175703627234,
            "Precision": 0.6941616069774302,
            "Recall": 0.6646754536723044,
            "F1-Score": 0.6790986121897534,
            "support": "46142226"
          }
        },
        "TIDE_errors": {
          "classification_error": 0.03301543970497283,
          "background_error": 0.0738072273682574,
          "missed_error": 0.09608140162059228
        },
        "BIoU": 0.0573697465297541
      }
    },
    "background": {
      "ungrouped": {
        "task_summary": {
          "mIoU": 0.6932138374376104,
          "mPrecision": 0.8333844579482547,
          "mRecall": 0.7858348301729027,
          "mF1-Score": 0.8076247284079171,
          "pixel_accuracy": 0.9049017903756111
        },
        "per_class": {
          "unlabeled": {
            "IoU": 0.8844649029535512,
            "Precision": 0.9259826887150658,
            "Recall": 0.9517525081044298,
            "F1-Score": 0.9386907674081014,
            "support": "604724222"
          },
          "background": {
            "IoU": 0.708119755640402,
            "Precision": 0.8575977296959723,
            "Recall": 0.8024763465806101,
            "F1-Score": 0.8291219082293401,
            "support": "186894437"
          },
          "dark": {
            "IoU": 0.4870568537188783,
            "Precision": 0.7165729554337263,
            "Recall": 0.6032756358336677,
            "F1-Score": 0.6550615095863097,
            "support": "30464925"
          }
        },
        "TIDE_errors": {
          "classification_error": 0.003645042983853083,
          "background_error": 0.03549082790102273,
          "missed_error": 0.055962338739511855
        },
        "BIoU": 0.09737677522435821
      }
    },
    "biota": {
      "ungrouped": {
        "task_summary": {
          "mIoU": 0.2874249292444779,
          "mPrecision": 0.4604675886853555,
          "mRecall": 0.40760183697248625,
          "mF1-Score": 0.4235191174642012,
          "pixel_accuracy": 0.992600835099511
        },
        "per_class": {
          "unlabeled": {
            "IoU": 0.9927889874353235,
            "Precision": 0.9960790187442637,
            "Recall": 0.996684058961475,
            "F1-Score": 0.9963814470020949,
            "support": "814728600"
          },
          "trash": {
            "IoU": 0.08593769251912159,
            "Precision": 0.24525831755477415,
            "Recall": 0.11683602598761439,
            "F1-Score": 0.15827370780319122,
            "support": "388031"
          },
          "other animal": {
            "IoU": 0.030887759230268597,
            "Precision": 0.06237444400153134,
            "Recall": 0.05765989273596264,
            "F1-Score": 0.05992458238776938,
            "support": "274090"
          },
          "millepora": {
            "IoU": 0.5925573283560835,
            "Precision": 0.7601101635008841,
            "Recall": 0.7288620964935624,
            "F1-Score": 0.744158238834329,
            "support": "3419710"
          },
          "clam": {
            "IoU": 0.2918585779191202,
            "Precision": 0.6867160356261268,
            "Recall": 0.33668775430842335,
            "F1-Score": 0.45184292291379996,
            "support": "527165"
          },
          "sea cucumber": {
            "IoU": 0.13578496786634936,
            "Precision": 0.2047236102902107,
            "Recall": 0.2873603673814851,
            "F1-Score": 0.2391033016072238,
            "support": "119331"
          },
          "turbinaria": {
            "IoU": 0.4894558583220436,
            "Precision": 0.5046741825130923,
            "Recall": 0.9419666376372179,
            "F1-Score": 0.6572277460755948,
            "support": "744072"
          },
          "sponge": {
            "IoU": 0.14283241923915488,
            "Precision": 0.3772495723515247,
            "Recall": 0.1869002871617678,
            "F1-Score": 0.24996214114094895,
            "support": "856660"
          },
          "anemone": {
            "IoU": 0.15038789065378258,
            "Precision": 0.26728253711685823,
            "Recall": 0.25587801243659913,
            "F1-Score": 0.2614559695483493,
            "support": "279176"
          },
          "sea urchin": {
            "IoU": 0.21395751595964838,
            "Precision": 0.5947857623847422,
            "Recall": 0.25046665535380963,
            "F1-Score": 0.35249588745370936,
            "support": "447868"
          },
          "crown of thorn": {
            "IoU": 0.32265015343283815,
            "Precision": 0.365889831454902,
            "Recall": 0.7319202552119171,
            "F1-Score": 0.48788434733920255,
            "support": "223814"
          },
          "dead clam": {
            "IoU": 0.0,
            "Precision": NaN,
            "Recall": 0.0,
            "F1-Score": NaN,
            "support": "75067"
          }
        },
        "TIDE_errors": {
          "classification_error": 0.00022462922699597392,
          "background_error": 0.0032862741120007533,
          "missed_error": 0.003888261561491047
        },
        "BIoU": 0.0740905873931519
      }
    }
  },
  "global_summary": {
    "task_summary": {
      "mIoU": 0.16136493142543773,
      "mPrecision": 0.5047545573996032,
      "mRecall": 0.24063690077105435,
      "mF1-Score": 0.4890250718821437,
      "pixel_accuracy": 0.2682792945783962
    },
    "per_class": {
      "background": {
        "IoU": 0.0,
        "Precision": NaN,
        "Recall": 0.0,
        "F1-Score": NaN,
        "support": "186894437"
      },
      "seagrass": {
        "IoU": 0.0,
        "Precision": NaN,
        "Recall": 0.0,
        "F1-Score": NaN,
        "support": "2892895"
      },
      "trash": {
        "IoU": 0.0,
        "Precision": NaN,
        "Recall": 0.0,
        "F1-Score": NaN,
        "support": "388031"
      },
      "other coral dead": {
        "IoU": 0.04108020002246522,
        "Precision": 0.23250223898950806,
        "Recall": 0.04752491776284588,
        "F1-Score": 0.07891841574084063,
        "support": "1054268"
      },
      "other coral bleached": {
        "IoU": 0.6829972163689594,
        "Precision": 0.7586522888911507,
        "Recall": 0.8725943360381841,
        "F1-Score": 0.8116439049644008,
        "support": "2434515"
      },
      "sand": {
        "IoU": 0.0,
        "Precision": NaN,
        "Recall": 0.0,
        "F1-Score": NaN,
        "support": "72597241"
      },
      "other coral alive": {
        "IoU": 0.33965265999551875,
        "Precision": 0.6306513632108687,
        "Recall": 0.423994338393843,
        "F1-Score": 0.5070757072160106,
        "support": "17700984"
      },
      "human": {
        "IoU": 0.0,
        "Precision": NaN,
        "Recall": 0.0,
        "F1-Score": NaN,
        "support": "14910111"
      },
      "transect tools": {
        "IoU": 0.0,
        "Precision": NaN,
        "Recall": 0.0,
        "F1-Score": NaN,
        "support": "1239260"
      },
      "fish": {
        "IoU": 0.0,
        "Precision": NaN,
        "Recall": 0.0,
        "F1-Score": NaN,
        "support": "5300718"
      },
      "algae covered substrate": {
        "IoU": 0.0,
        "Precision": NaN,
        "Recall": 0.0,
        "F1-Score": NaN,
        "support": "68937214"
      },
      "other animal": {
        "IoU": 0.0,
        "Precision": NaN,
        "Recall": 0.0,
        "F1-Score": NaN,
        "support": "274090"
      },
      "unknown hard substrate": {
        "IoU": 0.0,
        "Precision": NaN,
        "Recall": 0.0,
        "F1-Score": NaN,
        "support": "108065538"
      },
      "dark": {
        "IoU": 0.0,
        "Precision": NaN,
        "Recall": 0.0,
        "F1-Score": NaN,
        "support": "30464925"
      },
      "transect line": {
        "IoU": 0.0,
        "Precision": NaN,
        "Recall": 0.0,
        "F1-Score": NaN,
        "support": "3125978"
      },
      "massive/meandering bleached": {
        "IoU": 0.5319675378144917,
        "Precision": 0.846714405195113,
        "Recall": 0.5886583725222652,
        "F1-Score": 0.6944893082700666,
        "support": "5219586"
      },
      "massive/meandering alive": {
        "IoU": 0.6455463801553926,
        "Precision": 0.7206977659018177,
        "Recall": 0.8609325489838789,
        "F1-Score": 0.7845982197043054,
        "support": "40545023"
      },
      "rubble": {
        "IoU": 0.0,
        "Precision": NaN,
        "Recall": 0.0,
        "F1-Score": NaN,
        "support": "46142226"
      },
      "branching bleached": {
        "IoU": 0.4009727065802742,
        "Precision": 0.8035988333821826,
        "Recall": 0.4445366460710323,
        "F1-Score": 0.5724204400227535,
        "support": "1881143"
      },
      "branching dead": {
        "IoU": 0.36293842682887284,
        "Precision": 0.5534882502113706,
        "Recall": 0.5131981186941578,
        "F1-Score": 0.5325822791177968,
        "support": "8396083"
      },
      "millepora": {
        "IoU": 0.0,
        "Precision": NaN,
        "Recall": 0.0,
        "F1-Score": NaN,
        "support": "3419710"
      },
      "branching alive": {
        "IoU": 0.16021223868787102,
        "Precision": 0.21692751813038236,
        "Recall": 0.3799557084664142,
        "F1-Score": 0.27617746709698776,
        "support": "2044183"
      },
      "massive/meandering dead": {
        "IoU": 0.16564880828134038,
        "Precision": 0.3164807019902735,
        "Recall": 0.25792355749528817,
        "F1-Score": 0.28421735106575857,
        "support": "9399692"
      },
      "clam": {
        "IoU": 0.0,
        "Precision": NaN,
        "Recall": 0.0,
        "F1-Score": NaN,
        "support": "527165"
      },
      "acropora alive": {
        "IoU": 0.3606457248537872,
        "Precision": 0.6692010153511423,
        "Recall": 0.4388881317952042,
        "F1-Score": 0.5301096652363959,
        "support": "2649019"
      },
      "sea cucumber": {
        "IoU": 0.0,
        "Precision": NaN,
        "Recall": 0.0,
        "F1-Score": NaN,
        "support": "119331"
      },
      "turbinaria": {
        "IoU": 0.0,
        "Precision": NaN,
        "Recall": 0.0,
        "F1-Score": NaN,
        "support": "744072"
      },
      "table acropora alive": {
        "IoU": 0.2814184454370381,
        "Precision": 0.3513976395383826,
        "Recall": 0.5856006378269125,
        "F1-Score": 0.43922958412083424,
        "support": "1429855"
      },
      "sponge": {
        "IoU": 0.0,
        "Precision": NaN,
        "Recall": 0.0,
        "F1-Score": NaN,
        "support": "856660"
      },
      "anemone": {
        "IoU": 0.0,
        "Precision": NaN,
        "Recall": 0.0,
        "F1-Score": NaN,
        "support": "279176"
      },
      "pocillopora alive": {
        "IoU": 0.7537265141484235,
        "Precision": 0.8262687729432027,
        "Recall": 0.8956711269817604,
        "F1-Score": 0.8595713277613516,
        "support": "8169378"
      },
      "table acropora dead": {
        "IoU": 0.011562743989446194,
        "Precision": 0.014759446961056591,
        "Recall": 0.05068052447660757,
        "F1-Score": 0.022861150350090058,
        "support": "677933"
      },
      "meandering bleached": {
        "IoU": 0.4470871404681845,
        "Precision": 0.649296660880119,
        "Recall": 0.5894237173336344,
        "F1-Score": 0.6179132243874902,
        "support": "885850"
      },
      "stylophora alive": {
        "IoU": 0.2642777729086922,
        "Precision": 0.3170133872619606,
        "Recall": 0.6137015371809355,
        "F1-Score": 0.41806915943902884,
        "support": "437294"
      },
      "sea urchin": {
        "IoU": 0.0,
        "Precision": NaN,
        "Recall": 0.0,
        "F1-Score": NaN,
        "support": "447868"
      },
      "meandering alive": {
        "IoU": 0.6709683203890687,
        "Precision": 0.7893120026471997,
        "Recall": 0.8173559309517215,
        "F1-Score": 0.8030892174339251,
        "support": "13541250"
      },
      "meandering dead": {
        "IoU": 0.1337996874613621,
        "Precision": 0.1849170739690613,
        "Recall": 0.32615471651950156,
        "F1-Score": 0.23601997591117124,
        "support": "2591307"
      },
      "crown of thorn": {
        "IoU": 0.0,
        "Precision": NaN,
        "Recall": 0.0,
        "F1-Score": NaN,
        "support": "223814"
      },
      "dead clam": {
        "IoU": 0.0,
        "Precision": NaN,
        "Recall": 0.0,
        "F1-Score": NaN,
        "support": "75067"
      }
    },
    "TIDE_errors": {
      "classification_error": 0.038828816949104734,
      "background_error": 0.015342245296556105,
      "missed_error": 0.6775496431759418
    }
  },
  "optimization_metrics": {
    "tasks.genus.ungrouped.mIoU": 0.3988144926552631,
    "tasks.genus.ungrouped.BIoU": 0.11457209995613145,
    "tasks.genus.grouped.mIoU": 0.48643825570710064,
    "tasks.genus.grouped.BIoU": 0.11812449087697233,
    "tasks.health.ungrouped.mIoU": 0.4065487132236443,
    "tasks.health.ungrouped.BIoU": 0.12042858774130918,
    "tasks.health.grouped.mIoU": 0.6087960113717259,
    "tasks.health.grouped.BIoU": 0.1322200460455979,
    "tasks.fish.ungrouped.mIoU": 0.7650369900617493,
    "tasks.fish.ungrouped.BIoU": 0.19093821799256344,
    "tasks.human_artifacts.ungrouped.mIoU": 0.7273973028539837,
    "tasks.human_artifacts.ungrouped.BIoU": 0.37182436331301455,
    "tasks.substrate.ungrouped.mIoU": 0.555353888133594,
    "tasks.substrate.ungrouped.BIoU": 0.0573697465297541,
    "tasks.background.ungrouped.mIoU": 0.6932138374376104,
    "tasks.background.ungrouped.BIoU": 0.09737677522435821,
    "tasks.biota.ungrouped.mIoU": 0.2874249292444779,
    "tasks.biota.ungrouped.BIoU": 0.0740905873931519,
    "global.mIoU": 0.16136493142543773,
    "global.BIoU": 0.040097695566418554,
    "global.classification_error": 0.038828816949104734,
    "global.background_error": 0.015342245296556105,
    "global.missed_error": 0.6775496431759418,
    "global.Boundary_F1": 0.0771034049222036,
    "global.Boundary_Precision": 0.17413883308355554,
    "global.Boundary_Recall": 0.049513453089856926,
    "global.NLL": 12.512158393859863,
    "global.Brier_Score": 1.3596287965774536,
    "global.ECE": 0.6244975924754043,
    "test_weighted_genus_loss": 0.04143261221684135,
    "test_dwa_weight_genus": 0.141762837767601,
    "test_weighted_health_loss": 0.04148695658303189,
    "test_dwa_weight_health": 0.14172741770744324,
    "test_weighted_fish_loss": 0.0017240582291174643,
    "test_dwa_weight_fish": 0.1416475623846054,
    "test_weighted_human_artifacts_loss": 0.006332317542413021,
    "test_dwa_weight_human_artifacts": 0.14324988424777985,
    "test_weighted_substrate_loss": 0.07741921691566098,
    "test_dwa_weight_substrate": 0.14336620271205902,
    "test_weighted_background_loss": 0.03995464711773152,
    "test_dwa_weight_background": 0.14407768845558167,
    "test_weighted_biota_loss": 0.004474831311440818,
    "test_dwa_weight_biota": 0.144168421626091,
    "test_total_loss": 0.21282508969306946,
    "test_consistency_loss": 4.485858030032163e-06,
    "test_primary_balanced_loss": 0.08291956906834123,
    "test_aux_balanced_loss": 0.1299050704252963,
    "test_unweighted_genus_loss": 0.2922670904699029,
    "test_unweighted_health_loss": 0.2927235781538243,
    "test_unweighted_fish_loss": 0.012171464181048986,
    "test_unweighted_human_artifacts_loss": 0.044204695797370915,
    "test_unweighted_substrate_loss": 0.5400102333146699,
    "test_unweighted_background_loss": 0.27731321531595016,
    "test_unweighted_biota_loss": 0.03103891446502233,
    "test_unweighted_aux_loss_sum": 0.9047385174400953
  }
}
2025-09-28 23:50:57,331 - __main__ - INFO - [_log_key_metrics:359] -    Task-specific metrics:
2025-09-28 23:50:57,331 - __main__ - INFO - [_log_key_metrics:364] -      tasks: mIoU=N/A, BIoU=N/A
2025-09-28 23:50:57,331 - __main__ - DEBUG - [_log_key_metrics:365] -      tasks detailed metrics: {
  "genus": {
    "ungrouped": {
      "task_summary": {
        "mIoU": 0.3988144926552631,
        "mPrecision": 0.5469384216246881,
        "mRecall": 0.5369424366760911,
        "mF1-Score": 0.5238650500701681,
        "pixel_accuracy": 0.9143569566765599
      },
      "per_class": {
        "unlabeled": {
          "IoU": 0.9241583434035467,
          "Precision": 0.9630122237895932,
          "Recall": 0.9581689926754524,
          "F1-Score": 0.9605845034238185,
          "support": "703026221"
        },
        "other coral dead": {
          "IoU": 0.04108020002246522,
          "Precision": 0.23250223898950806,
          "Recall": 0.04752491776284588,
          "F1-Score": 0.07891841574084063,
          "support": "1054268"
        },
        "other coral bleached": {
          "IoU": 0.6829972163689594,
          "Precision": 0.7586522888911507,
          "Recall": 0.8725943360381841,
          "F1-Score": 0.8116439049644008,
          "support": "2434515"
        },
        "other coral alive": {
          "IoU": 0.33965265999551875,
          "Precision": 0.6306513632108687,
          "Recall": 0.423994338393843,
          "F1-Score": 0.5070757072160106,
          "support": "17700984"
        },
        "massive/meandering bleached": {
          "IoU": 0.5319675378144917,
          "Precision": 0.846714405195113,
          "Recall": 0.5886583725222652,
          "F1-Score": 0.6944893082700666,
          "support": "5219586"
        },
        "massive/meandering alive": {
          "IoU": 0.6455463801553926,
          "Precision": 0.7206977659018177,
          "Recall": 0.8609325489838789,
          "F1-Score": 0.7845982197043054,
          "support": "40545023"
        },
        "branching bleached": {
          "IoU": 0.4009727065802742,
          "Precision": 0.8035988333821826,
          "Recall": 0.4445366460710323,
          "F1-Score": 0.5724204400227535,
          "support": "1881143"
        },
        "branching dead": {
          "IoU": 0.36293842682887284,
          "Precision": 0.5534882502113706,
          "Recall": 0.5131981186941578,
          "F1-Score": 0.5325822791177968,
          "support": "8396083"
        },
        "branching alive": {
          "IoU": 0.16021223868787102,
          "Precision": 0.21692751813038236,
          "Recall": 0.3799557084664142,
          "F1-Score": 0.27617746709698776,
          "support": "2044183"
        },
        "massive/meandering dead": {
          "IoU": 0.16564880828134038,
          "Precision": 0.3164807019902735,
          "Recall": 0.25792355749528817,
          "F1-Score": 0.28421735106575857,
          "support": "9399692"
        },
        "acropora alive": {
          "IoU": 0.3606457248537872,
          "Precision": 0.6692010153511423,
          "Recall": 0.4388881317952042,
          "F1-Score": 0.5301096652363959,
          "support": "2649019"
        },
        "table acropora alive": {
          "IoU": 0.2814184454370381,
          "Precision": 0.3513976395383826,
          "Recall": 0.5856006378269125,
          "F1-Score": 0.43922958412083424,
          "support": "1429855"
        },
        "pocillopora alive": {
          "IoU": 0.7537265141484235,
          "Precision": 0.8262687729432027,
          "Recall": 0.8956711269817604,
          "F1-Score": 0.8595713277613516,
          "support": "8169378"
        },
        "table acropora dead": {
          "IoU": 0.011562743989446194,
          "Precision": 0.014759446961056591,
          "Recall": 0.05068052447660757,
          "F1-Score": 0.022861150350090058,
          "support": "677933"
        },
        "meandering bleached": {
          "IoU": 0.4470871404681845,
          "Precision": 0.649296660880119,
          "Recall": 0.5894237173336344,
          "F1-Score": 0.6179132243874902,
          "support": "885850"
        },
        "stylophora alive": {
          "IoU": 0.2642777729086922,
          "Precision": 0.3170133872619606,
          "Recall": 0.6137015371809355,
          "F1-Score": 0.41806915943902884,
          "support": "437294"
        },
        "meandering alive": {
          "IoU": 0.6709683203890687,
          "Precision": 0.7893120026471997,
          "Recall": 0.8173559309517215,
          "F1-Score": 0.8030892174339251,
          "support": "13541250"
        },
        "meandering dead": {
          "IoU": 0.1337996874613621,
          "Precision": 0.1849170739690613,
          "Recall": 0.32615471651950156,
          "F1-Score": 0.23601997591117124,
          "support": "2591307"
        }
      },
      "TIDE_errors": {
        "classification_error": 0.018398185165560955,
        "background_error": 0.03577287708009988,
        "missed_error": 0.03147198107777806
      },
      "BIoU": 0.11457209995613145
    },
    "grouped": {
      "task_summary": {
        "mIoU": 0.48643825570710064,
        "mPrecision": 0.6203186948535933,
        "mRecall": 0.6602646281880057,
        "mF1-Score": 0.6241139360863852,
        "pixel_accuracy": 0.9195085009750044
      },
      "per_class": {
        "unlabeled": {
          "IoU": 0.9241583434035467,
          "Precision": 0.9630122237895932,
          "Recall": 0.9581689926754524,
          "F1-Score": 0.9605845034238185,
          "support": "703026221"
        },
        "other coral": {
          "IoU": 0.38227386379606787,
          "Precision": 0.6694231265076694,
          "Recall": 0.471230854024964,
          "F1-Score": 0.5531087200712147,
          "support": "21189767"
        },
        "massive/meandering": {
          "IoU": 0.5969002320446193,
          "Precision": 0.7190384184972745,
          "Recall": 0.7784672373533746,
          "F1-Score": 0.7475736054974049,
          "support": "55164301"
        },
        "branching": {
          "IoU": 0.3609644022549157,
          "Precision": 0.5286459145693909,
          "Recall": 0.532274190394946,
          "F1-Score": 0.5304538482518005,
          "support": "12321409"
        },
        "acropora": {
          "IoU": 0.3606457248537872,
          "Precision": 0.6692010153511423,
          "Recall": 0.4388881317952042,
          "F1-Score": 0.5301096652363959,
          "support": "2649019"
        },
        "table acropora": {
          "IoU": 0.17779093249333547,
          "Precision": 0.2184962162768028,
          "Recall": 0.48831808512051494,
          "F1-Score": 0.3019057586340206,
          "support": "2107788"
        },
        "pocillopora": {
          "IoU": 0.7537265141484235,
          "Precision": 0.8262687729432027,
          "Recall": 0.8956711269817604,
          "F1-Score": 0.8595713277613516,
          "support": "8169378"
        },
        "meandering": {
          "IoU": 0.5572065154605177,
          "Precision": 0.6717691784853035,
          "Recall": 0.7656614981648987,
          "F1-Score": 0.7156488364624306,
          "support": "17018407"
        },
        "stylophora": {
          "IoU": 0.2642777729086922,
          "Precision": 0.3170133872619606,
          "Recall": 0.6137015371809355,
          "F1-Score": 0.41806915943902884,
          "support": "437294"
        }
      },
      "TIDE_errors": {
        "classification_error": 0.013246640867116485,
        "background_error": 0.03577287708009988,
        "missed_error": 0.03147198107777806
      },
      "BIoU": 0.11812449087697233
    }
  },
  "health": {
    "ungrouped": {
      "task_summary": {
        "mIoU": 0.4065487132236443,
        "mPrecision": 0.5457162972931795,
        "mRecall": 0.5471977627423236,
        "mF1-Score": 0.5314867472963161,
        "pixel_accuracy": 0.9155230315364127
      },
      "per_class": {
        "unlabeled": {
          "IoU": 0.9248986887142818,
          "Precision": 0.9630253693446081,
          "Recall": 0.9589518212294389,
          "F1-Score": 0.9609842784318787,
          "support": "703026221"
        },
        "other coral dead": {
          "IoU": 0.03883489244272628,
          "Precision": 0.1633075595884721,
          "Recall": 0.048481031388603275,
          "F1-Score": 0.07476624577252994,
          "support": "1054268"
        },
        "other coral bleached": {
          "IoU": 0.6805493160560803,
          "Precision": 0.7466850470613438,
          "Recall": 0.8848394854827347,
          "F1-Score": 0.8099129368642345,
          "support": "2434515"
        },
        "other coral alive": {
          "IoU": 0.3472651086688745,
          "Precision": 0.648104017454651,
          "Recall": 0.42795722542882364,
          "F1-Score": 0.5155111736130086,
          "support": "17700984"
        },
        "massive/meandering bleached": {
          "IoU": 0.5774920814398878,
          "Precision": 0.8366271656329679,
          "Recall": 0.6508933850309201,
          "F1-Score": 0.7321647927548013,
          "support": "5219586"
        },
        "massive/meandering alive": {
          "IoU": 0.6457471878031318,
          "Precision": 0.7219936860945086,
          "Recall": 0.8594461766614363,
          "F1-Score": 0.7847465182852588,
          "support": "40545023"
        },
        "branching bleached": {
          "IoU": 0.4272084737402712,
          "Precision": 0.789657007064667,
          "Recall": 0.4820659567082354,
          "F1-Score": 0.5986630287034244,
          "support": "1881143"
        },
        "branching dead": {
          "IoU": 0.38361980253299355,
          "Precision": 0.5858526110830811,
          "Recall": 0.526361876127237,
          "F1-Score": 0.5545162071700629,
          "support": "8396083"
        },
        "branching alive": {
          "IoU": 0.17034122765332774,
          "Precision": 0.22652024668705792,
          "Recall": 0.4071739174036767,
          "F1-Score": 0.291096688091356,
          "support": "2044183"
        },
        "massive/meandering dead": {
          "IoU": 0.16756126070706365,
          "Precision": 0.32113276427130694,
          "Recall": 0.2594713741684302,
          "F1-Score": 0.28702778405921103,
          "support": "9399692"
        },
        "acropora alive": {
          "IoU": 0.3377421029511601,
          "Precision": 0.6872443981301206,
          "Recall": 0.3990816977907671,
          "F1-Score": 0.5049435196904927,
          "support": "2649019"
        },
        "table acropora alive": {
          "IoU": 0.29171480165861746,
          "Precision": 0.3756084166741651,
          "Recall": 0.5663609247091488,
          "F1-Score": 0.4516706029597912,
          "support": "1429855"
        },
        "pocillopora alive": {
          "IoU": 0.7581355277369065,
          "Precision": 0.8300940263526514,
          "Recall": 0.8973901073007027,
          "F1-Score": 0.8624312696903267,
          "support": "8169378"
        },
        "table acropora dead": {
          "IoU": 0.009597897264398551,
          "Precision": 0.012748902166689556,
          "Recall": 0.03738127514075875,
          "F1-Score": 0.019013306763821448,
          "support": "677933"
        },
        "meandering bleached": {
          "IoU": 0.5036286379037831,
          "Precision": 0.631470697286747,
          "Recall": 0.7132742563639443,
          "F1-Score": 0.669884338736584,
          "support": "885850"
        },
        "stylophora alive": {
          "IoU": 0.27922253916401685,
          "Precision": 0.33561371029009346,
          "Recall": 0.6243145343864769,
          "F1-Score": 0.43655037433360305,
          "support": "437294"
        },
        "meandering alive": {
          "IoU": 0.6454032969135397,
          "Precision": 0.7619618890109324,
          "Recall": 0.8083961968060556,
          "F1-Score": 0.784492529125464,
          "support": "13541250"
        },
        "meandering dead": {
          "IoU": 0.12891399467453593,
          "Precision": 0.18524583708316805,
          "Recall": 0.2977184872344342,
          "F1-Score": 0.22838585628784172,
          "support": "2591307"
        }
      },
      "TIDE_errors": {
        "classification_error": 0.017887477972069545,
        "background_error": 0.035103420821987814,
        "missed_error": 0.03148606966952885
      },
      "BIoU": 0.12042858774130918
    },
    "grouped": {
      "task_summary": {
        "mIoU": 0.6087960113717259,
        "mPrecision": 0.7341945437532049,
        "mRecall": 0.7157528221037814,
        "mF1-Score": 0.723928701876609,
        "pixel_accuracy": 0.9254017630401912
      },
      "per_class": {
        "unlabeled": {
          "IoU": 0.9248986887142818,
          "Precision": 0.9630253693446081,
          "Recall": 0.9589518212294389,
          "F1-Score": 0.9609842784318787,
          "support": "703026221"
        },
        "alive coral": {
          "IoU": 0.6640067668944513,
          "Precision": 0.7770418146732841,
          "Recall": 0.8202929191268868,
          "F1-Score": 0.7980818108494743,
          "support": "86516986"
        },
        "bleached coral": {
          "IoU": 0.6161429860333889,
          "Precision": 0.8180834100623456,
          "Recall": 0.7139641001223096,
          "F1-Score": 0.7624857346881553,
          "support": "10421094"
        },
        "dead coral": {
          "IoU": 0.23013560384478174,
          "Precision": 0.3786275809325819,
          "Recall": 0.36980244793649053,
          "F1-Score": 0.37416298353692756,
          "support": "22119283"
        }
      },
      "TIDE_errors": {
        "classification_error": 0.008008746468290981,
        "background_error": 0.035103420821987814,
        "missed_error": 0.03148606966952885
      },
      "BIoU": 0.1322200460455979
    }
  },
  "fish": {
    "ungrouped": {
      "task_summary": {
        "mIoU": 0.7650369900617493,
        "mPrecision": 0.8668028447219019,
        "mRecall": 0.8294203132995214,
        "mF1-Score": 0.8470927648810136,
        "pixel_accuracy": 0.9962813014886805
      },
      "per_class": {
        "unlabeled": {
          "IoU": 0.9962653993177546,
          "Precision": 0.9977974119908697,
          "Recall": 0.9984612213939366,
          "F1-Score": 0.9981292063252102,
          "support": "816782866"
        },
        "fish": {
          "IoU": 0.5338085808057441,
          "Precision": 0.7358082774529342,
          "Recall": 0.6603794052051062,
          "F1-Score": 0.6960563234368169,
          "support": "5300718"
        }
      },
      "TIDE_errors": {
        "classification_error": 0.0,
        "background_error": 0.0015288567056461239,
        "missed_error": 0.0021898418056721563
      },
      "BIoU": 0.19093821799256344
    }
  },
  "human_artifacts": {
    "ungrouped": {
      "task_summary": {
        "mIoU": 0.7273973028539837,
        "mPrecision": 0.8749170149937497,
        "mRecall": 0.7947752139805654,
        "mF1-Score": 0.8320375689791639,
        "pixel_accuracy": 0.9923563271152719
      },
      "per_class": {
        "unlabeled": {
          "IoU": 0.992764667672694,
          "Precision": 0.9947919250468319,
          "Recall": 0.9979514821494077,
          "F1-Score": 0.9963691988097341,
          "support": "802808235"
        },
        "human": {
          "IoU": 0.6998269436139382,
          "Precision": 0.887670679283797,
          "Recall": 0.7678245990254533,
          "F1-Score": 0.8234096373670398,
          "support": "14910111"
        },
        "transect tools": {
          "IoU": 0.5376223822751542,
          "Precision": 0.7539393588791395,
          "Recall": 0.6520286299888643,
          "F1-Score": 0.6992905260388541,
          "support": "1239260"
        },
        "transect line": {
          "IoU": 0.6793752178541484,
          "Precision": 0.8632660967652301,
          "Recall": 0.7612961447585364,
          "F1-Score": 0.8090809137010277,
          "support": "3125978"
        }
      },
      "TIDE_errors": {
        "classification_error": 0.0005410715025298443,
        "background_error": 0.002000486364169994,
        "missed_error": 0.005102115018027163
      },
      "BIoU": 0.37182436331301455
    }
  },
  "substrate": {
    "ungrouped": {
      "task_summary": {
        "mIoU": 0.555353888133594,
        "mPrecision": 0.7353601763190145,
        "mRecall": 0.6802651448180504,
        "mF1-Score": 0.7042194144189277,
        "pixel_accuracy": 0.7970959313061763
      },
      "per_class": {
        "unlabeled": {
          "IoU": 0.7681699150046479,
          "Precision": 0.8542029933887497,
          "Recall": 0.8840846549804606,
          "F1-Score": 0.8688869870321583,
          "support": "523448470"
        },
        "seagrass": {
          "IoU": 0.47630140425884804,
          "Precision": 0.7404013877630311,
          "Recall": 0.5717905420003145,
          "F1-Score": 0.6452630917843866,
          "support": "2892895"
        },
        "sand": {
          "IoU": 0.7191698798110272,
          "Precision": 0.8357832655134482,
          "Recall": 0.8375141969926929,
          "F1-Score": 0.8366478359777674,
          "support": "72597241"
        },
        "algae covered substrate": {
          "IoU": 0.4638832460224314,
          "Precision": 0.724746587379196,
          "Recall": 0.5630877250130822,
          "F1-Score": 0.6337708246649655,
          "support": "68937214"
        },
        "unknown hard substrate": {
          "IoU": 0.3904813133418862,
          "Precision": 0.5628652168922318,
          "Recall": 0.5604382962494482,
          "F1-Score": 0.5616491348645346,
          "support": "108065538"
        },
        "rubble": {
          "IoU": 0.5141175703627234,
          "Precision": 0.6941616069774302,
          "Recall": 0.6646754536723044,
          "F1-Score": 0.6790986121897534,
          "support": "46142226"
        }
      },
      "TIDE_errors": {
        "classification_error": 0.03301543970497283,
        "background_error": 0.0738072273682574,
        "missed_error": 0.09608140162059228
      },
      "BIoU": 0.0573697465297541
    }
  },
  "background": {
    "ungrouped": {
      "task_summary": {
        "mIoU": 0.6932138374376104,
        "mPrecision": 0.8333844579482547,
        "mRecall": 0.7858348301729027,
        "mF1-Score": 0.8076247284079171,
        "pixel_accuracy": 0.9049017903756111
      },
      "per_class": {
        "unlabeled": {
          "IoU": 0.8844649029535512,
          "Precision": 0.9259826887150658,
          "Recall": 0.9517525081044298,
          "F1-Score": 0.9386907674081014,
          "support": "604724222"
        },
        "background": {
          "IoU": 0.708119755640402,
          "Precision": 0.8575977296959723,
          "Recall": 0.8024763465806101,
          "F1-Score": 0.8291219082293401,
          "support": "186894437"
        },
        "dark": {
          "IoU": 0.4870568537188783,
          "Precision": 0.7165729554337263,
          "Recall": 0.6032756358336677,
          "F1-Score": 0.6550615095863097,
          "support": "30464925"
        }
      },
      "TIDE_errors": {
        "classification_error": 0.003645042983853083,
        "background_error": 0.03549082790102273,
        "missed_error": 0.055962338739511855
      },
      "BIoU": 0.09737677522435821
    }
  },
  "biota": {
    "ungrouped": {
      "task_summary": {
        "mIoU": 0.2874249292444779,
        "mPrecision": 0.4604675886853555,
        "mRecall": 0.40760183697248625,
        "mF1-Score": 0.4235191174642012,
        "pixel_accuracy": 0.992600835099511
      },
      "per_class": {
        "unlabeled": {
          "IoU": 0.9927889874353235,
          "Precision": 0.9960790187442637,
          "Recall": 0.996684058961475,
          "F1-Score": 0.9963814470020949,
          "support": "814728600"
        },
        "trash": {
          "IoU": 0.08593769251912159,
          "Precision": 0.24525831755477415,
          "Recall": 0.11683602598761439,
          "F1-Score": 0.15827370780319122,
          "support": "388031"
        },
        "other animal": {
          "IoU": 0.030887759230268597,
          "Precision": 0.06237444400153134,
          "Recall": 0.05765989273596264,
          "F1-Score": 0.05992458238776938,
          "support": "274090"
        },
        "millepora": {
          "IoU": 0.5925573283560835,
          "Precision": 0.7601101635008841,
          "Recall": 0.7288620964935624,
          "F1-Score": 0.744158238834329,
          "support": "3419710"
        },
        "clam": {
          "IoU": 0.2918585779191202,
          "Precision": 0.6867160356261268,
          "Recall": 0.33668775430842335,
          "F1-Score": 0.45184292291379996,
          "support": "527165"
        },
        "sea cucumber": {
          "IoU": 0.13578496786634936,
          "Precision": 0.2047236102902107,
          "Recall": 0.2873603673814851,
          "F1-Score": 0.2391033016072238,
          "support": "119331"
        },
        "turbinaria": {
          "IoU": 0.4894558583220436,
          "Precision": 0.5046741825130923,
          "Recall": 0.9419666376372179,
          "F1-Score": 0.6572277460755948,
          "support": "744072"
        },
        "sponge": {
          "IoU": 0.14283241923915488,
          "Precision": 0.3772495723515247,
          "Recall": 0.1869002871617678,
          "F1-Score": 0.24996214114094895,
          "support": "856660"
        },
        "anemone": {
          "IoU": 0.15038789065378258,
          "Precision": 0.26728253711685823,
          "Recall": 0.25587801243659913,
          "F1-Score": 0.2614559695483493,
          "support": "279176"
        },
        "sea urchin": {
          "IoU": 0.21395751595964838,
          "Precision": 0.5947857623847422,
          "Recall": 0.25046665535380963,
          "F1-Score": 0.35249588745370936,
          "support": "447868"
        },
        "crown of thorn": {
          "IoU": 0.32265015343283815,
          "Precision": 0.365889831454902,
          "Recall": 0.7319202552119171,
          "F1-Score": 0.48788434733920255,
          "support": "223814"
        },
        "dead clam": {
          "IoU": 0.0,
          "Precision": NaN,
          "Recall": 0.0,
          "F1-Score": NaN,
          "support": "75067"
        }
      },
      "TIDE_errors": {
        "classification_error": 0.00022462922699597392,
        "background_error": 0.0032862741120007533,
        "missed_error": 0.003888261561491047
      },
      "BIoU": 0.0740905873931519
    }
  }
}
2025-09-28 23:50:57,332 - __main__ - INFO - [_log_key_metrics:364] -      global_summary: mIoU=N/A, BIoU=N/A
2025-09-28 23:50:57,332 - __main__ - DEBUG - [_log_key_metrics:365] -      global_summary detailed metrics: {
  "task_summary": {
    "mIoU": 0.16136493142543773,
    "mPrecision": 0.5047545573996032,
    "mRecall": 0.24063690077105435,
    "mF1-Score": 0.4890250718821437,
    "pixel_accuracy": 0.2682792945783962
  },
  "per_class": {
    "background": {
      "IoU": 0.0,
      "Precision": NaN,
      "Recall": 0.0,
      "F1-Score": NaN,
      "support": "186894437"
    },
    "seagrass": {
      "IoU": 0.0,
      "Precision": NaN,
      "Recall": 0.0,
      "F1-Score": NaN,
      "support": "2892895"
    },
    "trash": {
      "IoU": 0.0,
      "Precision": NaN,
      "Recall": 0.0,
      "F1-Score": NaN,
      "support": "388031"
    },
    "other coral dead": {
      "IoU": 0.04108020002246522,
      "Precision": 0.23250223898950806,
      "Recall": 0.04752491776284588,
      "F1-Score": 0.07891841574084063,
      "support": "1054268"
    },
    "other coral bleached": {
      "IoU": 0.6829972163689594,
      "Precision": 0.7586522888911507,
      "Recall": 0.8725943360381841,
      "F1-Score": 0.8116439049644008,
      "support": "2434515"
    },
    "sand": {
      "IoU": 0.0,
      "Precision": NaN,
      "Recall": 0.0,
      "F1-Score": NaN,
      "support": "72597241"
    },
    "other coral alive": {
      "IoU": 0.33965265999551875,
      "Precision": 0.6306513632108687,
      "Recall": 0.423994338393843,
      "F1-Score": 0.5070757072160106,
      "support": "17700984"
    },
    "human": {
      "IoU": 0.0,
      "Precision": NaN,
      "Recall": 0.0,
      "F1-Score": NaN,
      "support": "14910111"
    },
    "transect tools": {
      "IoU": 0.0,
      "Precision": NaN,
      "Recall": 0.0,
      "F1-Score": NaN,
      "support": "1239260"
    },
    "fish": {
      "IoU": 0.0,
      "Precision": NaN,
      "Recall": 0.0,
      "F1-Score": NaN,
      "support": "5300718"
    },
    "algae covered substrate": {
      "IoU": 0.0,
      "Precision": NaN,
      "Recall": 0.0,
      "F1-Score": NaN,
      "support": "68937214"
    },
    "other animal": {
      "IoU": 0.0,
      "Precision": NaN,
      "Recall": 0.0,
      "F1-Score": NaN,
      "support": "274090"
    },
    "unknown hard substrate": {
      "IoU": 0.0,
      "Precision": NaN,
      "Recall": 0.0,
      "F1-Score": NaN,
      "support": "108065538"
    },
    "dark": {
      "IoU": 0.0,
      "Precision": NaN,
      "Recall": 0.0,
      "F1-Score": NaN,
      "support": "30464925"
    },
    "transect line": {
      "IoU": 0.0,
      "Precision": NaN,
      "Recall": 0.0,
      "F1-Score": NaN,
      "support": "3125978"
    },
    "massive/meandering bleached": {
      "IoU": 0.5319675378144917,
      "Precision": 0.846714405195113,
      "Recall": 0.5886583725222652,
      "F1-Score": 0.6944893082700666,
      "support": "5219586"
    },
    "massive/meandering alive": {
      "IoU": 0.6455463801553926,
      "Precision": 0.7206977659018177,
      "Recall": 0.8609325489838789,
      "F1-Score": 0.7845982197043054,
      "support": "40545023"
    },
    "rubble": {
      "IoU": 0.0,
      "Precision": NaN,
      "Recall": 0.0,
      "F1-Score": NaN,
      "support": "46142226"
    },
    "branching bleached": {
      "IoU": 0.4009727065802742,
      "Precision": 0.8035988333821826,
      "Recall": 0.4445366460710323,
      "F1-Score": 0.5724204400227535,
      "support": "1881143"
    },
    "branching dead": {
      "IoU": 0.36293842682887284,
      "Precision": 0.5534882502113706,
      "Recall": 0.5131981186941578,
      "F1-Score": 0.5325822791177968,
      "support": "8396083"
    },
    "millepora": {
      "IoU": 0.0,
      "Precision": NaN,
      "Recall": 0.0,
      "F1-Score": NaN,
      "support": "3419710"
    },
    "branching alive": {
      "IoU": 0.16021223868787102,
      "Precision": 0.21692751813038236,
      "Recall": 0.3799557084664142,
      "F1-Score": 0.27617746709698776,
      "support": "2044183"
    },
    "massive/meandering dead": {
      "IoU": 0.16564880828134038,
      "Precision": 0.3164807019902735,
      "Recall": 0.25792355749528817,
      "F1-Score": 0.28421735106575857,
      "support": "9399692"
    },
    "clam": {
      "IoU": 0.0,
      "Precision": NaN,
      "Recall": 0.0,
      "F1-Score": NaN,
      "support": "527165"
    },
    "acropora alive": {
      "IoU": 0.3606457248537872,
      "Precision": 0.6692010153511423,
      "Recall": 0.4388881317952042,
      "F1-Score": 0.5301096652363959,
      "support": "2649019"
    },
    "sea cucumber": {
      "IoU": 0.0,
      "Precision": NaN,
      "Recall": 0.0,
      "F1-Score": NaN,
      "support": "119331"
    },
    "turbinaria": {
      "IoU": 0.0,
      "Precision": NaN,
      "Recall": 0.0,
      "F1-Score": NaN,
      "support": "744072"
    },
    "table acropora alive": {
      "IoU": 0.2814184454370381,
      "Precision": 0.3513976395383826,
      "Recall": 0.5856006378269125,
      "F1-Score": 0.43922958412083424,
      "support": "1429855"
    },
    "sponge": {
      "IoU": 0.0,
      "Precision": NaN,
      "Recall": 0.0,
      "F1-Score": NaN,
      "support": "856660"
    },
    "anemone": {
      "IoU": 0.0,
      "Precision": NaN,
      "Recall": 0.0,
      "F1-Score": NaN,
      "support": "279176"
    },
    "pocillopora alive": {
      "IoU": 0.7537265141484235,
      "Precision": 0.8262687729432027,
      "Recall": 0.8956711269817604,
      "F1-Score": 0.8595713277613516,
      "support": "8169378"
    },
    "table acropora dead": {
      "IoU": 0.011562743989446194,
      "Precision": 0.014759446961056591,
      "Recall": 0.05068052447660757,
      "F1-Score": 0.022861150350090058,
      "support": "677933"
    },
    "meandering bleached": {
      "IoU": 0.4470871404681845,
      "Precision": 0.649296660880119,
      "Recall": 0.5894237173336344,
      "F1-Score": 0.6179132243874902,
      "support": "885850"
    },
    "stylophora alive": {
      "IoU": 0.2642777729086922,
      "Precision": 0.3170133872619606,
      "Recall": 0.6137015371809355,
      "F1-Score": 0.41806915943902884,
      "support": "437294"
    },
    "sea urchin": {
      "IoU": 0.0,
      "Precision": NaN,
      "Recall": 0.0,
      "F1-Score": NaN,
      "support": "447868"
    },
    "meandering alive": {
      "IoU": 0.6709683203890687,
      "Precision": 0.7893120026471997,
      "Recall": 0.8173559309517215,
      "F1-Score": 0.8030892174339251,
      "support": "13541250"
    },
    "meandering dead": {
      "IoU": 0.1337996874613621,
      "Precision": 0.1849170739690613,
      "Recall": 0.32615471651950156,
      "F1-Score": 0.23601997591117124,
      "support": "2591307"
    },
    "crown of thorn": {
      "IoU": 0.0,
      "Precision": NaN,
      "Recall": 0.0,
      "F1-Score": NaN,
      "support": "223814"
    },
    "dead clam": {
      "IoU": 0.0,
      "Precision": NaN,
      "Recall": 0.0,
      "F1-Score": NaN,
      "support": "75067"
    }
  },
  "TIDE_errors": {
    "classification_error": 0.038828816949104734,
    "background_error": 0.015342245296556105,
    "missed_error": 0.6775496431759418
  }
}
2025-09-28 23:50:57,332 - __main__ - INFO - [_log_key_metrics:364] -      optimization_metrics: mIoU=N/A, BIoU=N/A
2025-09-28 23:50:57,332 - __main__ - DEBUG - [_log_key_metrics:365] -      optimization_metrics detailed metrics: {
  "tasks.genus.ungrouped.mIoU": 0.3988144926552631,
  "tasks.genus.ungrouped.BIoU": 0.11457209995613145,
  "tasks.genus.grouped.mIoU": 0.48643825570710064,
  "tasks.genus.grouped.BIoU": 0.11812449087697233,
  "tasks.health.ungrouped.mIoU": 0.4065487132236443,
  "tasks.health.ungrouped.BIoU": 0.12042858774130918,
  "tasks.health.grouped.mIoU": 0.6087960113717259,
  "tasks.health.grouped.BIoU": 0.1322200460455979,
  "tasks.fish.ungrouped.mIoU": 0.7650369900617493,
  "tasks.fish.ungrouped.BIoU": 0.19093821799256344,
  "tasks.human_artifacts.ungrouped.mIoU": 0.7273973028539837,
  "tasks.human_artifacts.ungrouped.BIoU": 0.37182436331301455,
  "tasks.substrate.ungrouped.mIoU": 0.555353888133594,
  "tasks.substrate.ungrouped.BIoU": 0.0573697465297541,
  "tasks.background.ungrouped.mIoU": 0.6932138374376104,
  "tasks.background.ungrouped.BIoU": 0.09737677522435821,
  "tasks.biota.ungrouped.mIoU": 0.2874249292444779,
  "tasks.biota.ungrouped.BIoU": 0.0740905873931519,
  "global.mIoU": 0.16136493142543773,
  "global.BIoU": 0.040097695566418554,
  "global.classification_error": 0.038828816949104734,
  "global.background_error": 0.015342245296556105,
  "global.missed_error": 0.6775496431759418,
  "global.Boundary_F1": 0.0771034049222036,
  "global.Boundary_Precision": 0.17413883308355554,
  "global.Boundary_Recall": 0.049513453089856926,
  "global.NLL": 12.512158393859863,
  "global.Brier_Score": 1.3596287965774536,
  "global.ECE": 0.6244975924754043,
  "test_weighted_genus_loss": 0.04143261221684135,
  "test_dwa_weight_genus": 0.141762837767601,
  "test_weighted_health_loss": 0.04148695658303189,
  "test_dwa_weight_health": 0.14172741770744324,
  "test_weighted_fish_loss": 0.0017240582291174643,
  "test_dwa_weight_fish": 0.1416475623846054,
  "test_weighted_human_artifacts_loss": 0.006332317542413021,
  "test_dwa_weight_human_artifacts": 0.14324988424777985,
  "test_weighted_substrate_loss": 0.07741921691566098,
  "test_dwa_weight_substrate": 0.14336620271205902,
  "test_weighted_background_loss": 0.03995464711773152,
  "test_dwa_weight_background": 0.14407768845558167,
  "test_weighted_biota_loss": 0.004474831311440818,
  "test_dwa_weight_biota": 0.144168421626091,
  "test_total_loss": 0.21282508969306946,
  "test_consistency_loss": 4.485858030032163e-06,
  "test_primary_balanced_loss": 0.08291956906834123,
  "test_aux_balanced_loss": 0.1299050704252963,
  "test_unweighted_genus_loss": 0.2922670904699029,
  "test_unweighted_health_loss": 0.2927235781538243,
  "test_unweighted_fish_loss": 0.012171464181048986,
  "test_unweighted_human_artifacts_loss": 0.044204695797370915,
  "test_unweighted_substrate_loss": 0.5400102333146699,
  "test_unweighted_background_loss": 0.27731321531595016,
  "test_unweighted_biota_loss": 0.03103891446502233,
  "test_unweighted_aux_loss_sum": 0.9047385174400953
}
2025-09-28 23:50:57,332 - __main__ - DEBUG - [_log_key_metrics:369] - Optimization metrics summary:
2025-09-28 23:50:57,332 - __main__ - DEBUG - [_log_key_metrics:371] -    tasks.genus.ungrouped.mIoU: 0.3988144926552631
2025-09-28 23:50:57,332 - __main__ - DEBUG - [_log_key_metrics:371] -    tasks.genus.ungrouped.BIoU: 0.11457209995613145
2025-09-28 23:50:57,332 - __main__ - DEBUG - [_log_key_metrics:371] -    tasks.genus.grouped.mIoU: 0.48643825570710064
2025-09-28 23:50:57,332 - __main__ - DEBUG - [_log_key_metrics:371] -    tasks.genus.grouped.BIoU: 0.11812449087697233
2025-09-28 23:50:57,332 - __main__ - DEBUG - [_log_key_metrics:371] -    tasks.health.ungrouped.mIoU: 0.4065487132236443
2025-09-28 23:50:57,332 - __main__ - DEBUG - [_log_key_metrics:371] -    tasks.health.ungrouped.BIoU: 0.12042858774130918
2025-09-28 23:50:57,332 - __main__ - DEBUG - [_log_key_metrics:371] -    tasks.health.grouped.mIoU: 0.6087960113717259
2025-09-28 23:50:57,332 - __main__ - DEBUG - [_log_key_metrics:371] -    tasks.health.grouped.BIoU: 0.1322200460455979
2025-09-28 23:50:57,332 - __main__ - DEBUG - [_log_key_metrics:371] -    tasks.fish.ungrouped.mIoU: 0.7650369900617493
2025-09-28 23:50:57,332 - __main__ - DEBUG - [_log_key_metrics:371] -    tasks.fish.ungrouped.BIoU: 0.19093821799256344
2025-09-28 23:50:57,332 - __main__ - DEBUG - [_log_key_metrics:371] -    tasks.human_artifacts.ungrouped.mIoU: 0.7273973028539837
2025-09-28 23:50:57,332 - __main__ - DEBUG - [_log_key_metrics:371] -    tasks.human_artifacts.ungrouped.BIoU: 0.37182436331301455
2025-09-28 23:50:57,332 - __main__ - DEBUG - [_log_key_metrics:371] -    tasks.substrate.ungrouped.mIoU: 0.555353888133594
2025-09-28 23:50:57,332 - __main__ - DEBUG - [_log_key_metrics:371] -    tasks.substrate.ungrouped.BIoU: 0.0573697465297541
2025-09-28 23:50:57,332 - __main__ - DEBUG - [_log_key_metrics:371] -    tasks.background.ungrouped.mIoU: 0.6932138374376104
2025-09-28 23:50:57,332 - __main__ - DEBUG - [_log_key_metrics:371] -    tasks.background.ungrouped.BIoU: 0.09737677522435821
2025-09-28 23:50:57,332 - __main__ - DEBUG - [_log_key_metrics:371] -    tasks.biota.ungrouped.mIoU: 0.2874249292444779
2025-09-28 23:50:57,332 - __main__ - DEBUG - [_log_key_metrics:371] -    tasks.biota.ungrouped.BIoU: 0.0740905873931519
2025-09-28 23:50:57,333 - __main__ - DEBUG - [_log_key_metrics:371] -    global.mIoU: 0.16136493142543773
2025-09-28 23:50:57,333 - __main__ - DEBUG - [_log_key_metrics:371] -    global.BIoU: 0.040097695566418554
2025-09-28 23:50:57,333 - __main__ - DEBUG - [_log_key_metrics:371] -    global.classification_error: 0.038828816949104734
2025-09-28 23:50:57,333 - __main__ - DEBUG - [_log_key_metrics:371] -    global.background_error: 0.015342245296556105
2025-09-28 23:50:57,333 - __main__ - DEBUG - [_log_key_metrics:371] -    global.missed_error: 0.6775496431759418
2025-09-28 23:50:57,333 - __main__ - DEBUG - [_log_key_metrics:371] -    global.Boundary_F1: 0.0771034049222036
2025-09-28 23:50:57,333 - __main__ - DEBUG - [_log_key_metrics:371] -    global.Boundary_Precision: 0.17413883308355554
2025-09-28 23:50:57,333 - __main__ - DEBUG - [_log_key_metrics:371] -    global.Boundary_Recall: 0.049513453089856926
2025-09-28 23:50:57,333 - __main__ - DEBUG - [_log_key_metrics:371] -    global.NLL: 12.512158393859863
2025-09-28 23:50:57,333 - __main__ - DEBUG - [_log_key_metrics:371] -    global.Brier_Score: 1.3596287965774536
2025-09-28 23:50:57,333 - __main__ - DEBUG - [_log_key_metrics:371] -    global.ECE: 0.6244975924754043
2025-09-28 23:50:57,333 - __main__ - DEBUG - [_log_key_metrics:371] -    test_weighted_genus_loss: 0.04143261221684135
2025-09-28 23:50:57,333 - __main__ - DEBUG - [_log_key_metrics:371] -    test_dwa_weight_genus: 0.141762837767601
2025-09-28 23:50:57,333 - __main__ - DEBUG - [_log_key_metrics:371] -    test_weighted_health_loss: 0.04148695658303189
2025-09-28 23:50:57,333 - __main__ - DEBUG - [_log_key_metrics:371] -    test_dwa_weight_health: 0.14172741770744324
2025-09-28 23:50:57,333 - __main__ - DEBUG - [_log_key_metrics:371] -    test_weighted_fish_loss: 0.0017240582291174643
2025-09-28 23:50:57,333 - __main__ - DEBUG - [_log_key_metrics:371] -    test_dwa_weight_fish: 0.1416475623846054
2025-09-28 23:50:57,333 - __main__ - DEBUG - [_log_key_metrics:371] -    test_weighted_human_artifacts_loss: 0.006332317542413021
2025-09-28 23:50:57,333 - __main__ - DEBUG - [_log_key_metrics:371] -    test_dwa_weight_human_artifacts: 0.14324988424777985
2025-09-28 23:50:57,333 - __main__ - DEBUG - [_log_key_metrics:371] -    test_weighted_substrate_loss: 0.07741921691566098
2025-09-28 23:50:57,333 - __main__ - DEBUG - [_log_key_metrics:371] -    test_dwa_weight_substrate: 0.14336620271205902
2025-09-28 23:50:57,333 - __main__ - DEBUG - [_log_key_metrics:371] -    test_weighted_background_loss: 0.03995464711773152
2025-09-28 23:50:57,333 - __main__ - DEBUG - [_log_key_metrics:371] -    test_dwa_weight_background: 0.14407768845558167
2025-09-28 23:50:57,333 - __main__ - DEBUG - [_log_key_metrics:371] -    test_weighted_biota_loss: 0.004474831311440818
2025-09-28 23:50:57,333 - __main__ - DEBUG - [_log_key_metrics:371] -    test_dwa_weight_biota: 0.144168421626091
2025-09-28 23:50:57,333 - __main__ - DEBUG - [_log_key_metrics:371] -    test_total_loss: 0.21282508969306946
2025-09-28 23:50:57,333 - __main__ - DEBUG - [_log_key_metrics:371] -    test_consistency_loss: 4.485858030032163e-06
2025-09-28 23:50:57,333 - __main__ - DEBUG - [_log_key_metrics:371] -    test_primary_balanced_loss: 0.08291956906834123
2025-09-28 23:50:57,333 - __main__ - DEBUG - [_log_key_metrics:371] -    test_aux_balanced_loss: 0.1299050704252963
2025-09-28 23:50:57,333 - __main__ - DEBUG - [_log_key_metrics:371] -    test_unweighted_genus_loss: 0.2922670904699029
2025-09-28 23:50:57,333 - __main__ - DEBUG - [_log_key_metrics:371] -    test_unweighted_health_loss: 0.2927235781538243
2025-09-28 23:50:57,333 - __main__ - DEBUG - [_log_key_metrics:371] -    test_unweighted_fish_loss: 0.012171464181048986
2025-09-28 23:50:57,333 - __main__ - DEBUG - [_log_key_metrics:371] -    test_unweighted_human_artifacts_loss: 0.044204695797370915
2025-09-28 23:50:57,333 - __main__ - DEBUG - [_log_key_metrics:371] -    test_unweighted_substrate_loss: 0.5400102333146699
2025-09-28 23:50:57,333 - __main__ - DEBUG - [_log_key_metrics:371] -    test_unweighted_background_loss: 0.27731321531595016
2025-09-28 23:50:57,333 - __main__ - DEBUG - [_log_key_metrics:371] -    test_unweighted_biota_loss: 0.03103891446502233
2025-09-28 23:50:57,333 - __main__ - DEBUG - [_log_key_metrics:371] -    test_unweighted_aux_loss_sum: 0.9047385174400953
2025-09-28 23:50:57,333 - __main__ - INFO - [run_single_experiment:301] - ✅ Evaluation completed successfully for mtl
2025-09-28 23:50:57,333 - __main__ - INFO - [run_single_experiment:311] - Total experiment duration: 2:50:36.018424
2025-09-28 23:50:57,333 - __main__ - INFO - [run_single_experiment:314] - 🎉 MTL experiment completed successfully!
2025-09-28 23:50:58,172 - __main__ - INFO - [run_comparison:433] - ✅ Experiment mtl completed successfully
2025-09-28 23:50:58,172 - __main__ - INFO - [run_comparison:408] - 
🚀 Running experiment 2/2: baseline
2025-09-28 23:50:58,172 - __main__ - DEBUG - [run_comparison:417] - Config path for baseline: /workspace/dbl_capstone/configs/baseline_comparisons/baseline_config.yaml
2025-09-28 23:50:58,172 - __main__ - DEBUG - [run_comparison:418] - Checkpoint path for baseline: None
2025-09-28 23:50:58,172 - __main__ - INFO - [run_single_experiment:211] - 
============================================================
2025-09-28 23:50:58,172 - __main__ - INFO - [run_single_experiment:212] - Starting BASELINE Experiment
2025-09-28 23:50:58,173 - __main__ - INFO - [run_single_experiment:213] - Config: /workspace/dbl_capstone/configs/baseline_comparisons/baseline_config.yaml
2025-09-28 23:50:58,173 - __main__ - INFO - [run_single_experiment:214] - Skip training: False
2025-09-28 23:50:58,173 - __main__ - INFO - [run_single_experiment:215] - Checkpoint path: Auto-detect best
2025-09-28 23:50:58,173 - __main__ - INFO - [run_single_experiment:216] - ============================================================
2025-09-28 23:50:58,173 - __main__ - DEBUG - [run_single_experiment:220] - Initializing ExperimentFactory with config: /workspace/dbl_capstone/configs/baseline_comparisons/baseline_config.yaml
2025-09-28 23:50:58,180 - __main__ - DEBUG - [run_single_experiment:222] - ExperimentFactory initialized successfully
2025-09-28 23:50:58,181 - __main__ - DEBUG - [run_single_experiment:237] - Logging experiment configuration details...
2025-09-28 23:50:58,181 - __main__ - INFO - [run_single_experiment:245] - Model type: SegFormerBaseline
2025-09-28 23:50:58,181 - __main__ - INFO - [run_single_experiment:246] - Primary tasks: []
2025-09-28 23:50:58,181 - __main__ - INFO - [run_single_experiment:247] - Auxiliary tasks: []
2025-09-28 23:50:58,181 - __main__ - INFO - [run_single_experiment:248] - Device policy: cuda
2025-09-28 23:50:58,181 - __main__ - INFO - [run_single_experiment:249] - Batch size per GPU: 8
2025-09-28 23:50:58,181 - __main__ - INFO - [run_single_experiment:250] - Learning rate: 6e-05
2025-09-28 23:50:58,181 - __main__ - INFO - [run_single_experiment:251] - Epochs: 50
2025-09-28 23:50:58,181 - __main__ - INFO - [run_single_experiment:252] - Output directory: /workspace/dbl_capstone/experiments/baseline_comparisons/coral_baseline_b2_run
2025-09-28 23:50:58,181 - __main__ - INFO - [run_single_experiment:258] - 🎯 Phase 1: Training & Validation for baseline
2025-09-28 23:50:58,181 - __main__ - DEBUG - [run_single_experiment:259] - Starting training phase...
2025-09-28 23:50:58,181 - __main__ - DEBUG - [run_single_experiment:262] - Training started at: 2025-09-28 23:50:58.181175
