2025-09-29 06:53:05,155 - __main__ - INFO - [_setup_logging:122] - Logging configured with file: /workspace/dbl_capstone/experiments/baselines_comparison/baseline_comparison_20250929_065305.md
2025-09-29 06:53:05,155 - __main__ - INFO - [_setup_logging:123] - Log level set to: DEBUG
2025-09-29 06:53:05,155 - __main__ - INFO - [_setup_logging:124] - Python version: 3.12.11 | packaged by conda-forge | (main, Jun  4 2025, 14:45:31) [GCC 13.3.0]
2025-09-29 06:53:05,155 - __main__ - INFO - [_setup_logging:125] - PyTorch version: 2.8.0+cu129
2025-09-29 06:53:05,155 - __main__ - INFO - [_setup_logging:126] - Current working directory: /workspace/dbl_capstone/experiments/baselines_comparison
2025-09-29 06:53:05,155 - __main__ - INFO - [_setup_logging:127] - Script directory: /workspace/dbl_capstone/experiments/baselines_comparison
2025-09-29 06:53:05,156 - __main__ - INFO - [_setup_logging:137] - Platform: Linux-6.8.0-83-generic-x86_64-with-glibc2.39
2025-09-29 06:53:05,157 - __main__ - INFO - [_setup_logging:138] - Architecture: ('64bit', '')
2025-09-29 06:53:05,303 - __main__ - INFO - [_setup_logging:140] - CUDA available: True, devices: 1
2025-09-29 06:53:05,306 - __main__ - INFO - [_setup_logging:142] -   GPU 0: NVIDIA GeForce RTX 5090
2025-09-29 06:53:05,306 - __main__ - DEBUG - [_validate_configs:180] - Starting configuration validation
2025-09-29 06:53:05,306 - __main__ - DEBUG - [_validate_configs:182] - Checking baseline config at path: /workspace/dbl_capstone/configs/baseline_comparisons/baseline_config.yaml
2025-09-29 06:53:05,306 - __main__ - DEBUG - [_validate_configs:189] - ✓ baseline config file exists
2025-09-29 06:53:05,306 - __main__ - DEBUG - [_validate_configs:182] - Checking mtl config at path: /workspace/dbl_capstone/configs/baseline_comparisons/mtl_config.yaml
2025-09-29 06:53:05,306 - __main__ - DEBUG - [_validate_configs:189] - ✓ mtl config file exists
2025-09-29 06:53:05,306 - __main__ - INFO - [_validate_configs:190] - All configuration files validated successfully
2025-09-29 06:53:05,306 - __main__ - INFO - [run_comparison:390] - 🔬 Starting Comprehensive Baseline Comparison
2025-09-29 06:53:05,306 - __main__ - INFO - [run_comparison:391] - Modes to run: mtl, baseline
2025-09-29 06:53:05,306 - __main__ - INFO - [run_comparison:392] - Skip training: False
2025-09-29 06:53:05,306 - __main__ - DEBUG - [run_comparison:393] - Config directory: /workspace/dbl_capstone/configs/baseline_comparisons
2025-09-29 06:53:05,306 - __main__ - DEBUG - [run_comparison:394] - Available config files: ['baseline', 'mtl']
2025-09-29 06:53:05,306 - __main__ - DEBUG - [run_comparison:404] - Checkpoint paths provided: {}
2025-09-29 06:53:05,306 - __main__ - INFO - [run_comparison:408] - 
🚀 Running experiment 1/2: mtl
2025-09-29 06:53:05,306 - __main__ - DEBUG - [run_comparison:417] - Config path for mtl: /workspace/dbl_capstone/configs/baseline_comparisons/mtl_config.yaml
2025-09-29 06:53:05,306 - __main__ - DEBUG - [run_comparison:418] - Checkpoint path for mtl: None
2025-09-29 06:53:05,306 - __main__ - INFO - [run_single_experiment:211] - 
============================================================
2025-09-29 06:53:05,306 - __main__ - INFO - [run_single_experiment:212] - Starting MTL Experiment
2025-09-29 06:53:05,306 - __main__ - INFO - [run_single_experiment:213] - Config: /workspace/dbl_capstone/configs/baseline_comparisons/mtl_config.yaml
2025-09-29 06:53:05,306 - __main__ - INFO - [run_single_experiment:214] - Skip training: False
2025-09-29 06:53:05,306 - __main__ - INFO - [run_single_experiment:215] - Checkpoint path: Auto-detect best
2025-09-29 06:53:05,306 - __main__ - INFO - [run_single_experiment:216] - ============================================================
2025-09-29 06:53:05,306 - __main__ - DEBUG - [run_single_experiment:220] - Initializing ExperimentFactory with config: /workspace/dbl_capstone/configs/baseline_comparisons/mtl_config.yaml
2025-09-29 06:53:05,314 - __main__ - DEBUG - [run_single_experiment:222] - ExperimentFactory initialized successfully
2025-09-29 06:53:05,314 - __main__ - DEBUG - [run_single_experiment:237] - Logging experiment configuration details...
2025-09-29 06:53:05,314 - __main__ - INFO - [run_single_experiment:245] - Model type: CoralMTL
2025-09-29 06:53:05,314 - __main__ - INFO - [run_single_experiment:246] - Primary tasks: ['genus', 'health']
2025-09-29 06:53:05,314 - __main__ - INFO - [run_single_experiment:247] - Auxiliary tasks: ['fish', 'human_artifacts', 'substrate', 'background', 'biota']
2025-09-29 06:53:05,314 - __main__ - INFO - [run_single_experiment:248] - Device policy: cuda
2025-09-29 06:53:05,314 - __main__ - INFO - [run_single_experiment:249] - Batch size per GPU: 4
2025-09-29 06:53:05,314 - __main__ - INFO - [run_single_experiment:250] - Learning rate: 6e-05
2025-09-29 06:53:05,314 - __main__ - INFO - [run_single_experiment:251] - Epochs: 50
2025-09-29 06:53:05,314 - __main__ - INFO - [run_single_experiment:252] - Output directory: /workspace/dbl_capstone/experiments/baseline_comparisons/coral_mtl_b2_run
2025-09-29 06:53:05,314 - __main__ - INFO - [run_single_experiment:258] - 🎯 Phase 1: Training & Validation for mtl
2025-09-29 06:53:05,314 - __main__ - DEBUG - [run_single_experiment:259] - Starting training phase...
2025-09-29 06:53:05,314 - __main__ - DEBUG - [run_single_experiment:262] - Training started at: 2025-09-29 06:53:05.314464
2025-09-29 09:27:13,845 - __main__ - INFO - [run_single_experiment:268] - Training duration: 2:34:08.530958
2025-09-29 09:27:13,846 - __main__ - INFO - [run_single_experiment:271] - ✅ Training completed successfully for mtl
2025-09-29 09:27:13,846 - __main__ - INFO - [run_single_experiment:283] - 🔍 Phase 2: Final Testing & Evaluation for mtl
2025-09-29 09:27:13,846 - __main__ - DEBUG - [run_single_experiment:284] - Starting evaluation phase...
2025-09-29 09:27:13,846 - __main__ - DEBUG - [run_single_experiment:287] - Evaluation started at: 2025-09-29 09:27:13.846164
2025-09-29 09:27:13,846 - __main__ - DEBUG - [run_single_experiment:288] - Using checkpoint: Auto-detect best
2025-09-29 09:41:31,096 - __main__ - INFO - [run_single_experiment:294] - Evaluation duration: 0:14:17.250168
2025-09-29 09:41:31,096 - __main__ - INFO - [_log_key_metrics:331] - 📊 Key Metrics for MTL:
2025-09-29 09:41:31,097 - __main__ - DEBUG - [_log_key_metrics:332] - Full metrics structure for mtl: {
  "tasks": {
    "genus": {
      "ungrouped": {
        "task_summary": {
          "mIoU": 0.18834475154771596,
          "mPrecision": 0.2420694445317991,
          "mRecall": 0.5641172857050208,
          "mF1-Score": 0.29472457033039495,
          "pixel_accuracy": 0.11005690024823546
        },
        "per_class": {
          "unlabeled": {
            "IoU": 0.0,
            "Precision": NaN,
            "Recall": 0.0,
            "F1-Score": NaN,
            "support": "703026221"
          },
          "other coral dead": {
            "IoU": 0.00310495398108945,
            "Precision": 0.0031329844957212107,
            "Recall": 0.25763278407387874,
            "F1-Score": 0.006190686166520487,
            "support": "1054268"
          },
          "other coral bleached": {
            "IoU": 0.2449772722201826,
            "Precision": 0.2523874122537402,
            "Recall": 0.8929778621203812,
            "F1-Score": 0.3935449709588863,
            "support": "2434515"
          },
          "other coral alive": {
            "IoU": 0.05217980040718575,
            "Precision": 0.05367910411605659,
            "Recall": 0.6513465014148366,
            "F1-Score": 0.09918418959761925,
            "support": "17700984"
          },
          "massive/meandering bleached": {
            "IoU": 0.4798537489166762,
            "Precision": 0.585219141940958,
            "Recall": 0.7271634187079206,
            "F1-Score": 0.6485150972086967,
            "support": "5219586"
          },
          "massive/meandering alive": {
            "IoU": 0.18098692197374577,
            "Precision": 0.18302314301512598,
            "Recall": 0.9420886751007639,
            "F1-Score": 0.30650114511220516,
            "support": "40545023"
          },
          "branching bleached": {
            "IoU": 0.2996937328169865,
            "Precision": 0.5611513697111211,
            "Recall": 0.391437014623556,
            "F1-Score": 0.4611759297591185,
            "support": "1881143"
          },
          "branching dead": {
            "IoU": 0.09789824720956099,
            "Precision": 0.10526644313187407,
            "Recall": 0.5830958317110491,
            "F1-Score": 0.17833755989388092,
            "support": "8396083"
          },
          "branching alive": {
            "IoU": 0.0621980704260504,
            "Precision": 0.06482872615836871,
            "Recall": 0.6051772272834672,
            "F1-Score": 0.11711200040328183,
            "support": "2044183"
          },
          "massive/meandering dead": {
            "IoU": 0.06192871043051485,
            "Precision": 0.06582594332415043,
            "Recall": 0.5112420704848627,
            "F1-Score": 0.11663440270940299,
            "support": "9399692"
          },
          "acropora alive": {
            "IoU": 0.22847990679770971,
            "Precision": 0.2677871924211761,
            "Recall": 0.6088487851540514,
            "F1-Score": 0.37197174415866596,
            "support": "2649019"
          },
          "table acropora alive": {
            "IoU": 0.020097145964765126,
            "Precision": 0.020212931322642874,
            "Recall": 0.7781921943134094,
            "F1-Score": 0.03940241582728494,
            "support": "1429855"
          },
          "pocillopora alive": {
            "IoU": 0.6742206235467618,
            "Precision": 0.7506831817760653,
            "Recall": 0.8687537778273939,
            "F1-Score": 0.8054143092783739,
            "support": "8169378"
          },
          "table acropora dead": {
            "IoU": 0.00439424722339318,
            "Precision": 0.004434298130295355,
            "Recall": 0.32728602974040205,
            "F1-Score": 0.008750044587652503,
            "support": "677933"
          },
          "meandering bleached": {
            "IoU": 0.22625879317754932,
            "Precision": 0.32405728355599267,
            "Recall": 0.42847773325055033,
            "F1-Score": 0.3690229084372232,
            "support": "885850"
          },
          "stylophora alive": {
            "IoU": 0.18521760442561536,
            "Precision": 0.24373449782553608,
            "Recall": 0.4354964852021752,
            "F1-Score": 0.3125461581637175,
            "support": "437294"
          },
          "meandering alive": {
            "IoU": 0.5211532321353824,
            "Precision": 0.5762547030684559,
            "Recall": 0.8449673405335549,
            "F1-Score": 0.6852080659931831,
            "support": "13541250"
          },
          "meandering dead": {
            "IoU": 0.04756251620571796,
            "Precision": 0.053502200793304415,
            "Recall": 0.29992741114811944,
            "F1-Score": 0.09080606736100082,
            "support": "2591307"
          }
        },
        "TIDE_errors": {
          "classification_error": 0.03476701463971815,
          "background_error": 0.8551760851120452,
          "missed_error": 0.0
        },
        "BIoU": 0.01579550759743983
      },
      "grouped": {
        "task_summary": {
          "mIoU": 0.19777907931966288,
          "mPrecision": 0.2500368730239964,
          "mRecall": 0.6336396012288977,
          "mF1-Score": 0.327638439469613,
          "pixel_accuracy": 0.11832457172627338
        },
        "per_class": {
          "unlabeled": {
            "IoU": 0.0,
            "Precision": NaN,
            "Recall": 0.0,
            "F1-Score": NaN,
            "support": "703026221"
          },
          "other coral": {
            "IoU": 0.04622401231903777,
            "Precision": 0.04720084502966793,
            "Recall": 0.6907431308706697,
            "F1-Score": 0.08836350872234067,
            "support": "21189767"
          },
          "massive/meandering": {
            "IoU": 0.1721653685216605,
            "Precision": 0.17499298386796763,
            "Recall": 0.9141985865097793,
            "F1-Score": 0.29375610838732785,
            "support": "55164301"
          },
          "branching": {
            "IoU": 0.11962788059970811,
            "Precision": 0.1265238890374216,
            "Recall": 0.6869974042741378,
            "F1-Score": 0.21369221448046047,
            "support": "12321409"
          },
          "acropora": {
            "IoU": 0.22847990679770971,
            "Precision": 0.2677871924211761,
            "Recall": 0.6088487851540514,
            "F1-Score": 0.37197174415866596,
            "support": "2649019"
          },
          "table acropora": {
            "IoU": 0.01424359511073161,
            "Precision": 0.01432524701696843,
            "Recall": 0.7141989611858498,
            "F1-Score": 0.028087128534790583,
            "support": "2107788"
          },
          "pocillopora": {
            "IoU": 0.6742206235467618,
            "Precision": 0.7506831817760653,
            "Recall": 0.8687537778273939,
            "F1-Score": 0.8054143092783739,
            "support": "8169378"
          },
          "meandering": {
            "IoU": 0.3398327225557409,
            "Precision": 0.37504714721716803,
            "Recall": 0.7835192800360222,
            "F1-Score": 0.5072763440312272,
            "support": "17018407"
          },
          "stylophora": {
            "IoU": 0.18521760442561536,
            "Precision": 0.24373449782553608,
            "Recall": 0.4354964852021752,
            "F1-Score": 0.3125461581637175,
            "support": "437294"
          }
        },
        "TIDE_errors": {
          "classification_error": 0.02649934316168023,
          "background_error": 0.8551760851120452,
          "missed_error": 0.0
        },
        "BIoU": 0.013585768511187648
      }
    },
    "health": {
      "ungrouped": {
        "task_summary": {
          "mIoU": 0.19625195611605628,
          "mPrecision": 0.25520344863406735,
          "mRecall": 0.5634012041399096,
          "mF1-Score": 0.30448807671537126,
          "pixel_accuracy": 0.10994216373988548
        },
        "per_class": {
          "unlabeled": {
            "IoU": 0.0,
            "Precision": NaN,
            "Recall": 0.0,
            "F1-Score": NaN,
            "support": "703026221"
          },
          "other coral dead": {
            "IoU": 0.0033868299701600036,
            "Precision": 0.003428832512786009,
            "Recall": 0.21659578020010092,
            "F1-Score": 0.006750796141625111,
            "support": "1054268"
          },
          "other coral bleached": {
            "IoU": 0.3274686143368797,
            "Precision": 0.34076936621814435,
            "Recall": 0.8935019911563494,
            "F1-Score": 0.4933730421949939,
            "support": "2434515"
          },
          "other coral alive": {
            "IoU": 0.04662619262828991,
            "Precision": 0.0479476313104624,
            "Recall": 0.6285019522078547,
            "F1-Score": 0.08909808097044107,
            "support": "17700984"
          },
          "massive/meandering bleached": {
            "IoU": 0.47373734641722115,
            "Precision": 0.5887372905305153,
            "Recall": 0.708052899214612,
            "F1-Score": 0.6429060749107245,
            "support": "5219586"
          },
          "massive/meandering alive": {
            "IoU": 0.18459140570297036,
            "Precision": 0.18666693617737617,
            "Recall": 0.9431869603329612,
            "F1-Score": 0.3116541362942416,
            "support": "40545023"
          },
          "branching bleached": {
            "IoU": 0.3057192701602234,
            "Precision": 0.5803778343075815,
            "Recall": 0.39247096047456254,
            "F1-Score": 0.46827718200514706,
            "support": "1881143"
          },
          "branching dead": {
            "IoU": 0.10161866897390547,
            "Precision": 0.1083518446773646,
            "Recall": 0.6205324554318961,
            "F1-Score": 0.18448973648668723,
            "support": "8396083"
          },
          "branching alive": {
            "IoU": 0.06024530211574839,
            "Precision": 0.06270648229725467,
            "Recall": 0.605513792062648,
            "F1-Score": 0.11364408216764038,
            "support": "2044183"
          },
          "massive/meandering dead": {
            "IoU": 0.06481457905748639,
            "Precision": 0.0689636742138551,
            "Recall": 0.5186076309734404,
            "F1-Score": 0.12173871457480714,
            "support": "9399692"
          },
          "acropora alive": {
            "IoU": 0.2382239162718294,
            "Precision": 0.28403230490246584,
            "Recall": 0.5963011212830108,
            "F1-Score": 0.38478325792494494,
            "support": "2649019"
          },
          "table acropora alive": {
            "IoU": 0.016501627525413624,
            "Precision": 0.01658786177149523,
            "Recall": 0.7604344496469921,
            "F1-Score": 0.03246748864649716,
            "support": "1429855"
          },
          "pocillopora alive": {
            "IoU": 0.6783244306134781,
            "Precision": 0.7601267774922192,
            "Recall": 0.8630728067669289,
            "F1-Score": 0.8083352875528721,
            "support": "8169378"
          },
          "table acropora dead": {
            "IoU": 0.005201460570257512,
            "Precision": 0.005246071868811257,
            "Recall": 0.37952423027054294,
            "F1-Score": 0.010349090753025146,
            "support": "677933"
          },
          "meandering bleached": {
            "IoU": 0.2765162410901439,
            "Precision": 0.41461914687248486,
            "Recall": 0.453602754416662,
            "F1-Score": 0.43323575868333525,
            "support": "885850"
          },
          "stylophora alive": {
            "IoU": 0.1687554103269982,
            "Precision": 0.22729027328086718,
            "Recall": 0.395870970102494,
            "F1-Score": 0.28877797499099195,
            "support": "437294"
          },
          "meandering alive": {
            "IoU": 0.533203762522365,
            "Precision": 0.5895809000883285,
            "Recall": 0.8479350872334533,
            "F1-Score": 0.6955419436815883,
            "support": "13541250"
          },
          "meandering dead": {
            "IoU": 0.0476001518056425,
            "Precision": 0.05302539825713323,
            "Recall": 0.31751583274386247,
            "F1-Score": 0.09087465618174821,
            "support": "2591307"
          }
        },
        "TIDE_errors": {
          "classification_error": 0.03488175114806814,
          "background_error": 0.8551760851120452,
          "missed_error": 0.0
        },
        "BIoU": 0.01528763137517262
      },
      "grouped": {
        "task_summary": {
          "mIoU": 0.15432595598393697,
          "mPrecision": 0.23441640019666002,
          "mRecall": 0.5740560862625359,
          "mF1-Score": 0.3155785144463942,
          "pixel_accuracy": 0.12694617315214493
        },
        "per_class": {
          "unlabeled": {
            "IoU": 0.0,
            "Precision": NaN,
            "Recall": 0.0,
            "F1-Score": NaN,
            "support": "703026221"
          },
          "alive coral": {
            "IoU": 0.14838762875219563,
            "Precision": 0.14927514778299045,
            "Recall": 0.9614760389364465,
            "F1-Score": 0.2584277730567845,
            "support": "86516986"
          },
          "bleached coral": {
            "IoU": 0.4157225930931831,
            "Precision": 0.49899711246231204,
            "Recall": 0.7135565613360747,
            "F1-Score": 0.5872938598583489,
            "support": "10421094"
          },
          "dead coral": {
            "IoU": 0.05319360209036912,
            "Precision": 0.05497694034467764,
            "Recall": 0.6211917447776223,
            "F1-Score": 0.1010139104240492,
            "support": "22119283"
          }
        },
        "TIDE_errors": {
          "classification_error": 0.017877741735808684,
          "background_error": 0.8551760851120452,
          "missed_error": 0.0
        },
        "BIoU": 0.014188954190884052
      }
    },
    "fish": {
      "ungrouped": {
        "task_summary": {
          "mIoU": 0.0032239531984134596,
          "mPrecision": 0.006447906396826919,
          "mRecall": 0.5,
          "mF1-Score": 0.012813194514778211,
          "pixel_accuracy": 0.006447906396826911
        },
        "per_class": {
          "unlabeled": {
            "IoU": 0.0,
            "Precision": NaN,
            "Recall": 0.0,
            "F1-Score": NaN,
            "support": "816782866"
          },
          "fish": {
            "IoU": 0.006447906396826919,
            "Precision": 0.006447906396826919,
            "Recall": 1.0,
            "F1-Score": 0.012813194514778211,
            "support": "5300718"
          }
        },
        "TIDE_errors": {
          "classification_error": 0.0,
          "background_error": 0.993552093603172,
          "missed_error": 0.0
        },
        "BIoU": 0.0
      }
    },
    "human_artifacts": {
      "ungrouped": {
        "task_summary": {
          "mIoU": 0.014265116919657985,
          "mPrecision": 0.01903650816642098,
          "mRecall": 0.6457901643031565,
          "mF1-Score": 0.036806443972625474,
          "pixel_accuracy": 0.02228336066615822
        },
        "per_class": {
          "unlabeled": {
            "IoU": 0.0,
            "Precision": NaN,
            "Recall": 0.0,
            "F1-Score": NaN,
            "support": "802808235"
          },
          "human": {
            "IoU": 0.042390352951283676,
            "Precision": 0.042416423448042063,
            "Recall": 0.9857078864134546,
            "F1-Score": 0.08133297249205221,
            "support": "14910111"
          },
          "transect tools": {
            "IoU": 0.0041289651410205425,
            "Precision": 0.004135373798340639,
            "Recall": 0.7270992366412213,
            "F1-Score": 0.008223973780978756,
            "support": "1239260"
          },
          "transect line": {
            "IoU": 0.010541149586327722,
            "Precision": 0.010557727252880227,
            "Recall": 0.8703535341579499,
            "F1-Score": 0.02086238564484547,
            "support": "3125978"
          }
        },
        "TIDE_errors": {
          "classification_error": 0.0011635848454066673,
          "background_error": 0.976553054488434,
          "missed_error": 0.0
        },
        "BIoU": 0.0035218195637992146
      }
    },
    "substrate": {
      "ungrouped": {
        "task_summary": {
          "mIoU": 0.32343909796358544,
          "mPrecision": 0.43947823922050216,
          "mRecall": 0.6580162073910047,
          "mF1-Score": 0.5434332766744048,
          "pixel_accuracy": 0.3032565761585621
        },
        "per_class": {
          "unlabeled": {
            "IoU": 0.0,
            "Precision": NaN,
            "Recall": 0.0,
            "F1-Score": NaN,
            "support": "523448470"
          },
          "seagrass": {
            "IoU": 0.2977078680511463,
            "Precision": 0.35317720812682574,
            "Recall": 0.6546397294060102,
            "F1-Score": 0.45882108813632133,
            "support": "2892895"
          },
          "sand": {
            "IoU": 0.5648203869915536,
            "Precision": 0.5780522835703872,
            "Recall": 0.9610514537322431,
            "F1-Score": 0.7218980423401171,
            "support": "72597241"
          },
          "algae covered substrate": {
            "IoU": 0.39206917984956213,
            "Precision": 0.4940723945082416,
            "Recall": 0.6550609950671926,
            "F1-Score": 0.5632897926695457,
            "support": "68937214"
          },
          "unknown hard substrate": {
            "IoU": 0.17305502463643674,
            "Precision": 0.17685490954533578,
            "Recall": 0.8895559378050752,
            "F1-Score": 0.29505014002232577,
            "support": "108065538"
          },
          "rubble": {
            "IoU": 0.512982128252814,
            "Precision": 0.5952344003517205,
            "Recall": 0.7877891283355077,
            "F1-Score": 0.6781073202037141,
            "support": "46142226"
          }
        },
        "TIDE_errors": {
          "classification_error": 0.06000954399303507,
          "background_error": 0.6367338798484017,
          "missed_error": 0.0
        },
        "BIoU": 0.033949123063802884
      }
    },
    "background": {
      "ungrouped": {
        "task_summary": {
          "mIoU": 0.15126913537551623,
          "mPrecision": 0.22917458462040702,
          "mRecall": 0.6354587698547371,
          "mF1-Score": 0.36796679894556905,
          "pixel_accuracy": 0.25555334529098167
        },
        "per_class": {
          "unlabeled": {
            "IoU": 0.0,
            "Precision": NaN,
            "Recall": 0.0,
            "F1-Score": NaN,
            "support": "604724222"
          },
          "background": {
            "IoU": 0.26891351619432313,
            "Precision": 0.27103323878324215,
            "Recall": 0.9717386237665276,
            "F1-Score": 0.42384845422852496,
            "support": "186894437"
          },
          "dark": {
            "IoU": 0.18489388993222552,
            "Precision": 0.18731593045757192,
            "Recall": 0.9346376857976837,
            "F1-Score": 0.3120851436626131,
            "support": "30464925"
          }
        },
        "TIDE_errors": {
          "classification_error": 0.00884721717055962,
          "background_error": 0.7355994375384575,
          "missed_error": 0.0
        },
        "BIoU": 0.020095228741868903
      }
    },
    "biota": {
      "ungrouped": {
        "task_summary": {
          "mIoU": 0.017823063759064815,
          "mPrecision": 0.01951702802586427,
          "mRecall": 0.7126409083385981,
          "mF1-Score": 0.03706003680081901,
          "pixel_accuracy": 0.007885520555535132
        },
        "per_class": {
          "unlabeled": {
            "IoU": 0.0,
            "Precision": NaN,
            "Recall": 0.0,
            "F1-Score": NaN,
            "support": "814728600"
          },
          "trash": {
            "IoU": 0.0015455667289280962,
            "Precision": 0.0015467531198747385,
            "Recall": 0.6683280459550912,
            "F1-Score": 0.003086363277461163,
            "support": "388031"
          },
          "other animal": {
            "IoU": 0.0013554337616689086,
            "Precision": 0.0013557309628754901,
            "Recall": 0.860782954503995,
            "F1-Score": 0.002707198095639462,
            "support": "274090"
          },
          "millepora": {
            "IoU": 0.05453469126445085,
            "Precision": 0.05458097289628837,
            "Recall": 0.9846893450029388,
            "F1-Score": 0.10342891839634116,
            "support": "3419710"
          },
          "clam": {
            "IoU": 0.024508254993033378,
            "Precision": 0.025034703386196365,
            "Recall": 0.5382053057391898,
            "F1-Score": 0.047843938540446475,
            "support": "527165"
          },
          "sea cucumber": {
            "IoU": 0.0038866119240086525,
            "Precision": 0.0038939656696419436,
            "Recall": 0.6729936060202294,
            "F1-Score": 0.007743129309314581,
            "support": "119331"
          },
          "turbinaria": {
            "IoU": 0.07993559225296497,
            "Precision": 0.07995583323694447,
            "Recall": 0.9968430474470212,
            "F1-Score": 0.14803770303783229,
            "support": "744072"
          },
          "sponge": {
            "IoU": 0.002623846361461447,
            "Precision": 0.0026251849963840235,
            "Recall": 0.8372820021945696,
            "F1-Score": 0.005233959617025726,
            "support": "856660"
          },
          "anemone": {
            "IoU": 0.012822676970760867,
            "Precision": 0.012983213036689716,
            "Recall": 0.5090874573745594,
            "F1-Score": 0.025320675103981793,
            "support": "279176"
          },
          "sea urchin": {
            "IoU": 0.011126891816830778,
            "Precision": 0.011143169041335346,
            "Recall": 0.8839546473514518,
            "F1-Score": 0.02200889306155741,
            "support": "447868"
          },
          "crown of thorn": {
            "IoU": 0.020486815137313347,
            "Precision": 0.020516844949461713,
            "Recall": 0.9333196314797109,
            "F1-Score": 0.04015106287200136,
            "support": "223814"
          },
          "dead clam": {
            "IoU": 0.0010503838973565107,
            "Precision": 0.0010509369888148236,
            "Recall": 0.6662048569944183,
            "F1-Score": 0.002098563497407764,
            "support": "75067"
          }
        },
        "TIDE_errors": {
          "classification_error": 0.0010612388046420337,
          "background_error": 0.9910532406398217,
          "missed_error": 0.0
        },
        "BIoU": 0.0020526994600347306
      }
    }
  },
  "global_summary": {
    "task_summary": {
      "mIoU": 0.08475513819647218,
      "mPrecision": 0.2420694445317991,
      "mRecall": 0.2538527785672593,
      "mF1-Score": 0.29472457033039495,
      "pixel_accuracy": 0.11005690024823546
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
        "IoU": 0.00310495398108945,
        "Precision": 0.0031329844957212107,
        "Recall": 0.25763278407387874,
        "F1-Score": 0.006190686166520487,
        "support": "1054268"
      },
      "other coral bleached": {
        "IoU": 0.2449772722201826,
        "Precision": 0.2523874122537402,
        "Recall": 0.8929778621203812,
        "F1-Score": 0.3935449709588863,
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
        "IoU": 0.05217980040718575,
        "Precision": 0.05367910411605659,
        "Recall": 0.6513465014148366,
        "F1-Score": 0.09918418959761925,
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
        "IoU": 0.4798537489166762,
        "Precision": 0.585219141940958,
        "Recall": 0.7271634187079206,
        "F1-Score": 0.6485150972086967,
        "support": "5219586"
      },
      "massive/meandering alive": {
        "IoU": 0.18098692197374577,
        "Precision": 0.18302314301512598,
        "Recall": 0.9420886751007639,
        "F1-Score": 0.30650114511220516,
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
        "IoU": 0.2996937328169865,
        "Precision": 0.5611513697111211,
        "Recall": 0.391437014623556,
        "F1-Score": 0.4611759297591185,
        "support": "1881143"
      },
      "branching dead": {
        "IoU": 0.09789824720956099,
        "Precision": 0.10526644313187407,
        "Recall": 0.5830958317110491,
        "F1-Score": 0.17833755989388092,
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
        "IoU": 0.0621980704260504,
        "Precision": 0.06482872615836871,
        "Recall": 0.6051772272834672,
        "F1-Score": 0.11711200040328183,
        "support": "2044183"
      },
      "massive/meandering dead": {
        "IoU": 0.06192871043051485,
        "Precision": 0.06582594332415043,
        "Recall": 0.5112420704848627,
        "F1-Score": 0.11663440270940299,
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
        "IoU": 0.22847990679770971,
        "Precision": 0.2677871924211761,
        "Recall": 0.6088487851540514,
        "F1-Score": 0.37197174415866596,
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
        "IoU": 0.020097145964765126,
        "Precision": 0.020212931322642874,
        "Recall": 0.7781921943134094,
        "F1-Score": 0.03940241582728494,
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
        "IoU": 0.6742206235467618,
        "Precision": 0.7506831817760653,
        "Recall": 0.8687537778273939,
        "F1-Score": 0.8054143092783739,
        "support": "8169378"
      },
      "table acropora dead": {
        "IoU": 0.00439424722339318,
        "Precision": 0.004434298130295355,
        "Recall": 0.32728602974040205,
        "F1-Score": 0.008750044587652503,
        "support": "677933"
      },
      "meandering bleached": {
        "IoU": 0.22625879317754932,
        "Precision": 0.32405728355599267,
        "Recall": 0.42847773325055033,
        "F1-Score": 0.3690229084372232,
        "support": "885850"
      },
      "stylophora alive": {
        "IoU": 0.18521760442561536,
        "Precision": 0.24373449782553608,
        "Recall": 0.4354964852021752,
        "F1-Score": 0.3125461581637175,
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
        "IoU": 0.5211532321353824,
        "Precision": 0.5762547030684559,
        "Recall": 0.8449673405335549,
        "F1-Score": 0.6852080659931831,
        "support": "13541250"
      },
      "meandering dead": {
        "IoU": 0.04756251620571796,
        "Precision": 0.053502200793304415,
        "Recall": 0.29992741114811944,
        "F1-Score": 0.09080606736100082,
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
      "classification_error": 0.7012753085214256,
      "background_error": 0.18866779123033775,
      "missed_error": 0.0
    }
  },
  "optimization_metrics": {
    "tasks.genus.ungrouped.mIoU": 0.18834475154771596,
    "tasks.genus.ungrouped.BIoU": 0.01579550759743983,
    "tasks.genus.grouped.mIoU": 0.19777907931966288,
    "tasks.genus.grouped.BIoU": 0.013585768511187648,
    "tasks.health.ungrouped.mIoU": 0.19625195611605628,
    "tasks.health.ungrouped.BIoU": 0.01528763137517262,
    "tasks.health.grouped.mIoU": 0.15432595598393697,
    "tasks.health.grouped.BIoU": 0.014188954190884052,
    "tasks.fish.ungrouped.mIoU": 0.0032239531984134596,
    "tasks.fish.ungrouped.BIoU": 0.0,
    "tasks.human_artifacts.ungrouped.mIoU": 0.014265116919657985,
    "tasks.human_artifacts.ungrouped.BIoU": 0.0035218195637992146,
    "tasks.substrate.ungrouped.mIoU": 0.32343909796358544,
    "tasks.substrate.ungrouped.BIoU": 0.033949123063802884,
    "tasks.background.ungrouped.mIoU": 0.15126913537551623,
    "tasks.background.ungrouped.BIoU": 0.020095228741868903,
    "tasks.biota.ungrouped.mIoU": 0.017823063759064815,
    "tasks.biota.ungrouped.BIoU": 0.0020526994600347306,
    "global.mIoU": 0.08475513819647218,
    "global.BIoU": 0.008225378539666952,
    "global.classification_error": 0.7012753085214256,
    "global.background_error": 0.18866779123033775,
    "global.missed_error": 0.0,
    "global.Boundary_F1": 0.016316058072420907,
    "global.Boundary_Precision": 0.013766088558529734,
    "global.Boundary_Recall": 0.020026946532500103,
    "global.NLL": 14.44483470916748,
    "global.Brier_Score": 1.1155517101287842,
    "global.ECE": 0.3203518580493579,
    "test_weighted_genus_loss": 0.0594511532167695,
    "test_dwa_weight_genus": 0.13594737648963928,
    "test_weighted_health_loss": 0.059546656830578436,
    "test_dwa_weight_health": 0.13598665595054626,
    "test_weighted_fish_loss": 2.10917737069473e-09,
    "test_dwa_weight_fish": 0.18656256794929504,
    "test_weighted_human_artifacts_loss": 0.027037290191605726,
    "test_dwa_weight_human_artifacts": 0.1313881278038025,
    "test_weighted_substrate_loss": 0.07152267786370096,
    "test_dwa_weight_substrate": 0.13596321642398834,
    "test_weighted_background_loss": 0.013573172810482698,
    "test_dwa_weight_background": 0.13595537841320038,
    "test_weighted_biota_loss": 0.08825631391426919,
    "test_dwa_weight_biota": 0.13819670677185059,
    "test_total_loss": 0.31938726584217986,
    "test_consistency_loss": 0.0,
    "test_primary_balanced_loss": 0.11899781025642035,
    "test_aux_balanced_loss": 0.20038945792356924,
    "test_unweighted_genus_loss": 0.4373100415176275,
    "test_unweighted_health_loss": 0.4378860282654665,
    "test_unweighted_fish_loss": 1.1305469272501388e-08,
    "test_unweighted_human_artifacts_loss": 0.20578183637627837,
    "test_unweighted_substrate_loss": 0.5260443221397546,
    "test_unweighted_background_loss": 0.0998354976894619,
    "test_unweighted_biota_loss": 0.6386282006373667,
    "test_unweighted_aux_loss_sum": 1.470289869728137
  }
}
2025-09-29 09:41:31,098 - __main__ - INFO - [_log_key_metrics:359] -    Task-specific metrics:
2025-09-29 09:41:31,098 - __main__ - INFO - [_log_key_metrics:364] -      tasks: mIoU=N/A, BIoU=N/A
2025-09-29 09:41:31,099 - __main__ - DEBUG - [_log_key_metrics:365] -      tasks detailed metrics: {
  "genus": {
    "ungrouped": {
      "task_summary": {
        "mIoU": 0.18834475154771596,
        "mPrecision": 0.2420694445317991,
        "mRecall": 0.5641172857050208,
        "mF1-Score": 0.29472457033039495,
        "pixel_accuracy": 0.11005690024823546
      },
      "per_class": {
        "unlabeled": {
          "IoU": 0.0,
          "Precision": NaN,
          "Recall": 0.0,
          "F1-Score": NaN,
          "support": "703026221"
        },
        "other coral dead": {
          "IoU": 0.00310495398108945,
          "Precision": 0.0031329844957212107,
          "Recall": 0.25763278407387874,
          "F1-Score": 0.006190686166520487,
          "support": "1054268"
        },
        "other coral bleached": {
          "IoU": 0.2449772722201826,
          "Precision": 0.2523874122537402,
          "Recall": 0.8929778621203812,
          "F1-Score": 0.3935449709588863,
          "support": "2434515"
        },
        "other coral alive": {
          "IoU": 0.05217980040718575,
          "Precision": 0.05367910411605659,
          "Recall": 0.6513465014148366,
          "F1-Score": 0.09918418959761925,
          "support": "17700984"
        },
        "massive/meandering bleached": {
          "IoU": 0.4798537489166762,
          "Precision": 0.585219141940958,
          "Recall": 0.7271634187079206,
          "F1-Score": 0.6485150972086967,
          "support": "5219586"
        },
        "massive/meandering alive": {
          "IoU": 0.18098692197374577,
          "Precision": 0.18302314301512598,
          "Recall": 0.9420886751007639,
          "F1-Score": 0.30650114511220516,
          "support": "40545023"
        },
        "branching bleached": {
          "IoU": 0.2996937328169865,
          "Precision": 0.5611513697111211,
          "Recall": 0.391437014623556,
          "F1-Score": 0.4611759297591185,
          "support": "1881143"
        },
        "branching dead": {
          "IoU": 0.09789824720956099,
          "Precision": 0.10526644313187407,
          "Recall": 0.5830958317110491,
          "F1-Score": 0.17833755989388092,
          "support": "8396083"
        },
        "branching alive": {
          "IoU": 0.0621980704260504,
          "Precision": 0.06482872615836871,
          "Recall": 0.6051772272834672,
          "F1-Score": 0.11711200040328183,
          "support": "2044183"
        },
        "massive/meandering dead": {
          "IoU": 0.06192871043051485,
          "Precision": 0.06582594332415043,
          "Recall": 0.5112420704848627,
          "F1-Score": 0.11663440270940299,
          "support": "9399692"
        },
        "acropora alive": {
          "IoU": 0.22847990679770971,
          "Precision": 0.2677871924211761,
          "Recall": 0.6088487851540514,
          "F1-Score": 0.37197174415866596,
          "support": "2649019"
        },
        "table acropora alive": {
          "IoU": 0.020097145964765126,
          "Precision": 0.020212931322642874,
          "Recall": 0.7781921943134094,
          "F1-Score": 0.03940241582728494,
          "support": "1429855"
        },
        "pocillopora alive": {
          "IoU": 0.6742206235467618,
          "Precision": 0.7506831817760653,
          "Recall": 0.8687537778273939,
          "F1-Score": 0.8054143092783739,
          "support": "8169378"
        },
        "table acropora dead": {
          "IoU": 0.00439424722339318,
          "Precision": 0.004434298130295355,
          "Recall": 0.32728602974040205,
          "F1-Score": 0.008750044587652503,
          "support": "677933"
        },
        "meandering bleached": {
          "IoU": 0.22625879317754932,
          "Precision": 0.32405728355599267,
          "Recall": 0.42847773325055033,
          "F1-Score": 0.3690229084372232,
          "support": "885850"
        },
        "stylophora alive": {
          "IoU": 0.18521760442561536,
          "Precision": 0.24373449782553608,
          "Recall": 0.4354964852021752,
          "F1-Score": 0.3125461581637175,
          "support": "437294"
        },
        "meandering alive": {
          "IoU": 0.5211532321353824,
          "Precision": 0.5762547030684559,
          "Recall": 0.8449673405335549,
          "F1-Score": 0.6852080659931831,
          "support": "13541250"
        },
        "meandering dead": {
          "IoU": 0.04756251620571796,
          "Precision": 0.053502200793304415,
          "Recall": 0.29992741114811944,
          "F1-Score": 0.09080606736100082,
          "support": "2591307"
        }
      },
      "TIDE_errors": {
        "classification_error": 0.03476701463971815,
        "background_error": 0.8551760851120452,
        "missed_error": 0.0
      },
      "BIoU": 0.01579550759743983
    },
    "grouped": {
      "task_summary": {
        "mIoU": 0.19777907931966288,
        "mPrecision": 0.2500368730239964,
        "mRecall": 0.6336396012288977,
        "mF1-Score": 0.327638439469613,
        "pixel_accuracy": 0.11832457172627338
      },
      "per_class": {
        "unlabeled": {
          "IoU": 0.0,
          "Precision": NaN,
          "Recall": 0.0,
          "F1-Score": NaN,
          "support": "703026221"
        },
        "other coral": {
          "IoU": 0.04622401231903777,
          "Precision": 0.04720084502966793,
          "Recall": 0.6907431308706697,
          "F1-Score": 0.08836350872234067,
          "support": "21189767"
        },
        "massive/meandering": {
          "IoU": 0.1721653685216605,
          "Precision": 0.17499298386796763,
          "Recall": 0.9141985865097793,
          "F1-Score": 0.29375610838732785,
          "support": "55164301"
        },
        "branching": {
          "IoU": 0.11962788059970811,
          "Precision": 0.1265238890374216,
          "Recall": 0.6869974042741378,
          "F1-Score": 0.21369221448046047,
          "support": "12321409"
        },
        "acropora": {
          "IoU": 0.22847990679770971,
          "Precision": 0.2677871924211761,
          "Recall": 0.6088487851540514,
          "F1-Score": 0.37197174415866596,
          "support": "2649019"
        },
        "table acropora": {
          "IoU": 0.01424359511073161,
          "Precision": 0.01432524701696843,
          "Recall": 0.7141989611858498,
          "F1-Score": 0.028087128534790583,
          "support": "2107788"
        },
        "pocillopora": {
          "IoU": 0.6742206235467618,
          "Precision": 0.7506831817760653,
          "Recall": 0.8687537778273939,
          "F1-Score": 0.8054143092783739,
          "support": "8169378"
        },
        "meandering": {
          "IoU": 0.3398327225557409,
          "Precision": 0.37504714721716803,
          "Recall": 0.7835192800360222,
          "F1-Score": 0.5072763440312272,
          "support": "17018407"
        },
        "stylophora": {
          "IoU": 0.18521760442561536,
          "Precision": 0.24373449782553608,
          "Recall": 0.4354964852021752,
          "F1-Score": 0.3125461581637175,
          "support": "437294"
        }
      },
      "TIDE_errors": {
        "classification_error": 0.02649934316168023,
        "background_error": 0.8551760851120452,
        "missed_error": 0.0
      },
      "BIoU": 0.013585768511187648
    }
  },
  "health": {
    "ungrouped": {
      "task_summary": {
        "mIoU": 0.19625195611605628,
        "mPrecision": 0.25520344863406735,
        "mRecall": 0.5634012041399096,
        "mF1-Score": 0.30448807671537126,
        "pixel_accuracy": 0.10994216373988548
      },
      "per_class": {
        "unlabeled": {
          "IoU": 0.0,
          "Precision": NaN,
          "Recall": 0.0,
          "F1-Score": NaN,
          "support": "703026221"
        },
        "other coral dead": {
          "IoU": 0.0033868299701600036,
          "Precision": 0.003428832512786009,
          "Recall": 0.21659578020010092,
          "F1-Score": 0.006750796141625111,
          "support": "1054268"
        },
        "other coral bleached": {
          "IoU": 0.3274686143368797,
          "Precision": 0.34076936621814435,
          "Recall": 0.8935019911563494,
          "F1-Score": 0.4933730421949939,
          "support": "2434515"
        },
        "other coral alive": {
          "IoU": 0.04662619262828991,
          "Precision": 0.0479476313104624,
          "Recall": 0.6285019522078547,
          "F1-Score": 0.08909808097044107,
          "support": "17700984"
        },
        "massive/meandering bleached": {
          "IoU": 0.47373734641722115,
          "Precision": 0.5887372905305153,
          "Recall": 0.708052899214612,
          "F1-Score": 0.6429060749107245,
          "support": "5219586"
        },
        "massive/meandering alive": {
          "IoU": 0.18459140570297036,
          "Precision": 0.18666693617737617,
          "Recall": 0.9431869603329612,
          "F1-Score": 0.3116541362942416,
          "support": "40545023"
        },
        "branching bleached": {
          "IoU": 0.3057192701602234,
          "Precision": 0.5803778343075815,
          "Recall": 0.39247096047456254,
          "F1-Score": 0.46827718200514706,
          "support": "1881143"
        },
        "branching dead": {
          "IoU": 0.10161866897390547,
          "Precision": 0.1083518446773646,
          "Recall": 0.6205324554318961,
          "F1-Score": 0.18448973648668723,
          "support": "8396083"
        },
        "branching alive": {
          "IoU": 0.06024530211574839,
          "Precision": 0.06270648229725467,
          "Recall": 0.605513792062648,
          "F1-Score": 0.11364408216764038,
          "support": "2044183"
        },
        "massive/meandering dead": {
          "IoU": 0.06481457905748639,
          "Precision": 0.0689636742138551,
          "Recall": 0.5186076309734404,
          "F1-Score": 0.12173871457480714,
          "support": "9399692"
        },
        "acropora alive": {
          "IoU": 0.2382239162718294,
          "Precision": 0.28403230490246584,
          "Recall": 0.5963011212830108,
          "F1-Score": 0.38478325792494494,
          "support": "2649019"
        },
        "table acropora alive": {
          "IoU": 0.016501627525413624,
          "Precision": 0.01658786177149523,
          "Recall": 0.7604344496469921,
          "F1-Score": 0.03246748864649716,
          "support": "1429855"
        },
        "pocillopora alive": {
          "IoU": 0.6783244306134781,
          "Precision": 0.7601267774922192,
          "Recall": 0.8630728067669289,
          "F1-Score": 0.8083352875528721,
          "support": "8169378"
        },
        "table acropora dead": {
          "IoU": 0.005201460570257512,
          "Precision": 0.005246071868811257,
          "Recall": 0.37952423027054294,
          "F1-Score": 0.010349090753025146,
          "support": "677933"
        },
        "meandering bleached": {
          "IoU": 0.2765162410901439,
          "Precision": 0.41461914687248486,
          "Recall": 0.453602754416662,
          "F1-Score": 0.43323575868333525,
          "support": "885850"
        },
        "stylophora alive": {
          "IoU": 0.1687554103269982,
          "Precision": 0.22729027328086718,
          "Recall": 0.395870970102494,
          "F1-Score": 0.28877797499099195,
          "support": "437294"
        },
        "meandering alive": {
          "IoU": 0.533203762522365,
          "Precision": 0.5895809000883285,
          "Recall": 0.8479350872334533,
          "F1-Score": 0.6955419436815883,
          "support": "13541250"
        },
        "meandering dead": {
          "IoU": 0.0476001518056425,
          "Precision": 0.05302539825713323,
          "Recall": 0.31751583274386247,
          "F1-Score": 0.09087465618174821,
          "support": "2591307"
        }
      },
      "TIDE_errors": {
        "classification_error": 0.03488175114806814,
        "background_error": 0.8551760851120452,
        "missed_error": 0.0
      },
      "BIoU": 0.01528763137517262
    },
    "grouped": {
      "task_summary": {
        "mIoU": 0.15432595598393697,
        "mPrecision": 0.23441640019666002,
        "mRecall": 0.5740560862625359,
        "mF1-Score": 0.3155785144463942,
        "pixel_accuracy": 0.12694617315214493
      },
      "per_class": {
        "unlabeled": {
          "IoU": 0.0,
          "Precision": NaN,
          "Recall": 0.0,
          "F1-Score": NaN,
          "support": "703026221"
        },
        "alive coral": {
          "IoU": 0.14838762875219563,
          "Precision": 0.14927514778299045,
          "Recall": 0.9614760389364465,
          "F1-Score": 0.2584277730567845,
          "support": "86516986"
        },
        "bleached coral": {
          "IoU": 0.4157225930931831,
          "Precision": 0.49899711246231204,
          "Recall": 0.7135565613360747,
          "F1-Score": 0.5872938598583489,
          "support": "10421094"
        },
        "dead coral": {
          "IoU": 0.05319360209036912,
          "Precision": 0.05497694034467764,
          "Recall": 0.6211917447776223,
          "F1-Score": 0.1010139104240492,
          "support": "22119283"
        }
      },
      "TIDE_errors": {
        "classification_error": 0.017877741735808684,
        "background_error": 0.8551760851120452,
        "missed_error": 0.0
      },
      "BIoU": 0.014188954190884052
    }
  },
  "fish": {
    "ungrouped": {
      "task_summary": {
        "mIoU": 0.0032239531984134596,
        "mPrecision": 0.006447906396826919,
        "mRecall": 0.5,
        "mF1-Score": 0.012813194514778211,
        "pixel_accuracy": 0.006447906396826911
      },
      "per_class": {
        "unlabeled": {
          "IoU": 0.0,
          "Precision": NaN,
          "Recall": 0.0,
          "F1-Score": NaN,
          "support": "816782866"
        },
        "fish": {
          "IoU": 0.006447906396826919,
          "Precision": 0.006447906396826919,
          "Recall": 1.0,
          "F1-Score": 0.012813194514778211,
          "support": "5300718"
        }
      },
      "TIDE_errors": {
        "classification_error": 0.0,
        "background_error": 0.993552093603172,
        "missed_error": 0.0
      },
      "BIoU": 0.0
    }
  },
  "human_artifacts": {
    "ungrouped": {
      "task_summary": {
        "mIoU": 0.014265116919657985,
        "mPrecision": 0.01903650816642098,
        "mRecall": 0.6457901643031565,
        "mF1-Score": 0.036806443972625474,
        "pixel_accuracy": 0.02228336066615822
      },
      "per_class": {
        "unlabeled": {
          "IoU": 0.0,
          "Precision": NaN,
          "Recall": 0.0,
          "F1-Score": NaN,
          "support": "802808235"
        },
        "human": {
          "IoU": 0.042390352951283676,
          "Precision": 0.042416423448042063,
          "Recall": 0.9857078864134546,
          "F1-Score": 0.08133297249205221,
          "support": "14910111"
        },
        "transect tools": {
          "IoU": 0.0041289651410205425,
          "Precision": 0.004135373798340639,
          "Recall": 0.7270992366412213,
          "F1-Score": 0.008223973780978756,
          "support": "1239260"
        },
        "transect line": {
          "IoU": 0.010541149586327722,
          "Precision": 0.010557727252880227,
          "Recall": 0.8703535341579499,
          "F1-Score": 0.02086238564484547,
          "support": "3125978"
        }
      },
      "TIDE_errors": {
        "classification_error": 0.0011635848454066673,
        "background_error": 0.976553054488434,
        "missed_error": 0.0
      },
      "BIoU": 0.0035218195637992146
    }
  },
  "substrate": {
    "ungrouped": {
      "task_summary": {
        "mIoU": 0.32343909796358544,
        "mPrecision": 0.43947823922050216,
        "mRecall": 0.6580162073910047,
        "mF1-Score": 0.5434332766744048,
        "pixel_accuracy": 0.3032565761585621
      },
      "per_class": {
        "unlabeled": {
          "IoU": 0.0,
          "Precision": NaN,
          "Recall": 0.0,
          "F1-Score": NaN,
          "support": "523448470"
        },
        "seagrass": {
          "IoU": 0.2977078680511463,
          "Precision": 0.35317720812682574,
          "Recall": 0.6546397294060102,
          "F1-Score": 0.45882108813632133,
          "support": "2892895"
        },
        "sand": {
          "IoU": 0.5648203869915536,
          "Precision": 0.5780522835703872,
          "Recall": 0.9610514537322431,
          "F1-Score": 0.7218980423401171,
          "support": "72597241"
        },
        "algae covered substrate": {
          "IoU": 0.39206917984956213,
          "Precision": 0.4940723945082416,
          "Recall": 0.6550609950671926,
          "F1-Score": 0.5632897926695457,
          "support": "68937214"
        },
        "unknown hard substrate": {
          "IoU": 0.17305502463643674,
          "Precision": 0.17685490954533578,
          "Recall": 0.8895559378050752,
          "F1-Score": 0.29505014002232577,
          "support": "108065538"
        },
        "rubble": {
          "IoU": 0.512982128252814,
          "Precision": 0.5952344003517205,
          "Recall": 0.7877891283355077,
          "F1-Score": 0.6781073202037141,
          "support": "46142226"
        }
      },
      "TIDE_errors": {
        "classification_error": 0.06000954399303507,
        "background_error": 0.6367338798484017,
        "missed_error": 0.0
      },
      "BIoU": 0.033949123063802884
    }
  },
  "background": {
    "ungrouped": {
      "task_summary": {
        "mIoU": 0.15126913537551623,
        "mPrecision": 0.22917458462040702,
        "mRecall": 0.6354587698547371,
        "mF1-Score": 0.36796679894556905,
        "pixel_accuracy": 0.25555334529098167
      },
      "per_class": {
        "unlabeled": {
          "IoU": 0.0,
          "Precision": NaN,
          "Recall": 0.0,
          "F1-Score": NaN,
          "support": "604724222"
        },
        "background": {
          "IoU": 0.26891351619432313,
          "Precision": 0.27103323878324215,
          "Recall": 0.9717386237665276,
          "F1-Score": 0.42384845422852496,
          "support": "186894437"
        },
        "dark": {
          "IoU": 0.18489388993222552,
          "Precision": 0.18731593045757192,
          "Recall": 0.9346376857976837,
          "F1-Score": 0.3120851436626131,
          "support": "30464925"
        }
      },
      "TIDE_errors": {
        "classification_error": 0.00884721717055962,
        "background_error": 0.7355994375384575,
        "missed_error": 0.0
      },
      "BIoU": 0.020095228741868903
    }
  },
  "biota": {
    "ungrouped": {
      "task_summary": {
        "mIoU": 0.017823063759064815,
        "mPrecision": 0.01951702802586427,
        "mRecall": 0.7126409083385981,
        "mF1-Score": 0.03706003680081901,
        "pixel_accuracy": 0.007885520555535132
      },
      "per_class": {
        "unlabeled": {
          "IoU": 0.0,
          "Precision": NaN,
          "Recall": 0.0,
          "F1-Score": NaN,
          "support": "814728600"
        },
        "trash": {
          "IoU": 0.0015455667289280962,
          "Precision": 0.0015467531198747385,
          "Recall": 0.6683280459550912,
          "F1-Score": 0.003086363277461163,
          "support": "388031"
        },
        "other animal": {
          "IoU": 0.0013554337616689086,
          "Precision": 0.0013557309628754901,
          "Recall": 0.860782954503995,
          "F1-Score": 0.002707198095639462,
          "support": "274090"
        },
        "millepora": {
          "IoU": 0.05453469126445085,
          "Precision": 0.05458097289628837,
          "Recall": 0.9846893450029388,
          "F1-Score": 0.10342891839634116,
          "support": "3419710"
        },
        "clam": {
          "IoU": 0.024508254993033378,
          "Precision": 0.025034703386196365,
          "Recall": 0.5382053057391898,
          "F1-Score": 0.047843938540446475,
          "support": "527165"
        },
        "sea cucumber": {
          "IoU": 0.0038866119240086525,
          "Precision": 0.0038939656696419436,
          "Recall": 0.6729936060202294,
          "F1-Score": 0.007743129309314581,
          "support": "119331"
        },
        "turbinaria": {
          "IoU": 0.07993559225296497,
          "Precision": 0.07995583323694447,
          "Recall": 0.9968430474470212,
          "F1-Score": 0.14803770303783229,
          "support": "744072"
        },
        "sponge": {
          "IoU": 0.002623846361461447,
          "Precision": 0.0026251849963840235,
          "Recall": 0.8372820021945696,
          "F1-Score": 0.005233959617025726,
          "support": "856660"
        },
        "anemone": {
          "IoU": 0.012822676970760867,
          "Precision": 0.012983213036689716,
          "Recall": 0.5090874573745594,
          "F1-Score": 0.025320675103981793,
          "support": "279176"
        },
        "sea urchin": {
          "IoU": 0.011126891816830778,
          "Precision": 0.011143169041335346,
          "Recall": 0.8839546473514518,
          "F1-Score": 0.02200889306155741,
          "support": "447868"
        },
        "crown of thorn": {
          "IoU": 0.020486815137313347,
          "Precision": 0.020516844949461713,
          "Recall": 0.9333196314797109,
          "F1-Score": 0.04015106287200136,
          "support": "223814"
        },
        "dead clam": {
          "IoU": 0.0010503838973565107,
          "Precision": 0.0010509369888148236,
          "Recall": 0.6662048569944183,
          "F1-Score": 0.002098563497407764,
          "support": "75067"
        }
      },
      "TIDE_errors": {
        "classification_error": 0.0010612388046420337,
        "background_error": 0.9910532406398217,
        "missed_error": 0.0
      },
      "BIoU": 0.0020526994600347306
    }
  }
}
2025-09-29 09:41:31,099 - __main__ - INFO - [_log_key_metrics:364] -      global_summary: mIoU=N/A, BIoU=N/A
2025-09-29 09:41:31,099 - __main__ - DEBUG - [_log_key_metrics:365] -      global_summary detailed metrics: {
  "task_summary": {
    "mIoU": 0.08475513819647218,
    "mPrecision": 0.2420694445317991,
    "mRecall": 0.2538527785672593,
    "mF1-Score": 0.29472457033039495,
    "pixel_accuracy": 0.11005690024823546
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
      "IoU": 0.00310495398108945,
      "Precision": 0.0031329844957212107,
      "Recall": 0.25763278407387874,
      "F1-Score": 0.006190686166520487,
      "support": "1054268"
    },
    "other coral bleached": {
      "IoU": 0.2449772722201826,
      "Precision": 0.2523874122537402,
      "Recall": 0.8929778621203812,
      "F1-Score": 0.3935449709588863,
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
      "IoU": 0.05217980040718575,
      "Precision": 0.05367910411605659,
      "Recall": 0.6513465014148366,
      "F1-Score": 0.09918418959761925,
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
      "IoU": 0.4798537489166762,
      "Precision": 0.585219141940958,
      "Recall": 0.7271634187079206,
      "F1-Score": 0.6485150972086967,
      "support": "5219586"
    },
    "massive/meandering alive": {
      "IoU": 0.18098692197374577,
      "Precision": 0.18302314301512598,
      "Recall": 0.9420886751007639,
      "F1-Score": 0.30650114511220516,
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
      "IoU": 0.2996937328169865,
      "Precision": 0.5611513697111211,
      "Recall": 0.391437014623556,
      "F1-Score": 0.4611759297591185,
      "support": "1881143"
    },
    "branching dead": {
      "IoU": 0.09789824720956099,
      "Precision": 0.10526644313187407,
      "Recall": 0.5830958317110491,
      "F1-Score": 0.17833755989388092,
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
      "IoU": 0.0621980704260504,
      "Precision": 0.06482872615836871,
      "Recall": 0.6051772272834672,
      "F1-Score": 0.11711200040328183,
      "support": "2044183"
    },
    "massive/meandering dead": {
      "IoU": 0.06192871043051485,
      "Precision": 0.06582594332415043,
      "Recall": 0.5112420704848627,
      "F1-Score": 0.11663440270940299,
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
      "IoU": 0.22847990679770971,
      "Precision": 0.2677871924211761,
      "Recall": 0.6088487851540514,
      "F1-Score": 0.37197174415866596,
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
      "IoU": 0.020097145964765126,
      "Precision": 0.020212931322642874,
      "Recall": 0.7781921943134094,
      "F1-Score": 0.03940241582728494,
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
      "IoU": 0.6742206235467618,
      "Precision": 0.7506831817760653,
      "Recall": 0.8687537778273939,
      "F1-Score": 0.8054143092783739,
      "support": "8169378"
    },
    "table acropora dead": {
      "IoU": 0.00439424722339318,
      "Precision": 0.004434298130295355,
      "Recall": 0.32728602974040205,
      "F1-Score": 0.008750044587652503,
      "support": "677933"
    },
    "meandering bleached": {
      "IoU": 0.22625879317754932,
      "Precision": 0.32405728355599267,
      "Recall": 0.42847773325055033,
      "F1-Score": 0.3690229084372232,
      "support": "885850"
    },
    "stylophora alive": {
      "IoU": 0.18521760442561536,
      "Precision": 0.24373449782553608,
      "Recall": 0.4354964852021752,
      "F1-Score": 0.3125461581637175,
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
      "IoU": 0.5211532321353824,
      "Precision": 0.5762547030684559,
      "Recall": 0.8449673405335549,
      "F1-Score": 0.6852080659931831,
      "support": "13541250"
    },
    "meandering dead": {
      "IoU": 0.04756251620571796,
      "Precision": 0.053502200793304415,
      "Recall": 0.29992741114811944,
      "F1-Score": 0.09080606736100082,
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
    "classification_error": 0.7012753085214256,
    "background_error": 0.18866779123033775,
    "missed_error": 0.0
  }
}
2025-09-29 09:41:31,100 - __main__ - INFO - [_log_key_metrics:364] -      optimization_metrics: mIoU=N/A, BIoU=N/A
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:365] -      optimization_metrics detailed metrics: {
  "tasks.genus.ungrouped.mIoU": 0.18834475154771596,
  "tasks.genus.ungrouped.BIoU": 0.01579550759743983,
  "tasks.genus.grouped.mIoU": 0.19777907931966288,
  "tasks.genus.grouped.BIoU": 0.013585768511187648,
  "tasks.health.ungrouped.mIoU": 0.19625195611605628,
  "tasks.health.ungrouped.BIoU": 0.01528763137517262,
  "tasks.health.grouped.mIoU": 0.15432595598393697,
  "tasks.health.grouped.BIoU": 0.014188954190884052,
  "tasks.fish.ungrouped.mIoU": 0.0032239531984134596,
  "tasks.fish.ungrouped.BIoU": 0.0,
  "tasks.human_artifacts.ungrouped.mIoU": 0.014265116919657985,
  "tasks.human_artifacts.ungrouped.BIoU": 0.0035218195637992146,
  "tasks.substrate.ungrouped.mIoU": 0.32343909796358544,
  "tasks.substrate.ungrouped.BIoU": 0.033949123063802884,
  "tasks.background.ungrouped.mIoU": 0.15126913537551623,
  "tasks.background.ungrouped.BIoU": 0.020095228741868903,
  "tasks.biota.ungrouped.mIoU": 0.017823063759064815,
  "tasks.biota.ungrouped.BIoU": 0.0020526994600347306,
  "global.mIoU": 0.08475513819647218,
  "global.BIoU": 0.008225378539666952,
  "global.classification_error": 0.7012753085214256,
  "global.background_error": 0.18866779123033775,
  "global.missed_error": 0.0,
  "global.Boundary_F1": 0.016316058072420907,
  "global.Boundary_Precision": 0.013766088558529734,
  "global.Boundary_Recall": 0.020026946532500103,
  "global.NLL": 14.44483470916748,
  "global.Brier_Score": 1.1155517101287842,
  "global.ECE": 0.3203518580493579,
  "test_weighted_genus_loss": 0.0594511532167695,
  "test_dwa_weight_genus": 0.13594737648963928,
  "test_weighted_health_loss": 0.059546656830578436,
  "test_dwa_weight_health": 0.13598665595054626,
  "test_weighted_fish_loss": 2.10917737069473e-09,
  "test_dwa_weight_fish": 0.18656256794929504,
  "test_weighted_human_artifacts_loss": 0.027037290191605726,
  "test_dwa_weight_human_artifacts": 0.1313881278038025,
  "test_weighted_substrate_loss": 0.07152267786370096,
  "test_dwa_weight_substrate": 0.13596321642398834,
  "test_weighted_background_loss": 0.013573172810482698,
  "test_dwa_weight_background": 0.13595537841320038,
  "test_weighted_biota_loss": 0.08825631391426919,
  "test_dwa_weight_biota": 0.13819670677185059,
  "test_total_loss": 0.31938726584217986,
  "test_consistency_loss": 0.0,
  "test_primary_balanced_loss": 0.11899781025642035,
  "test_aux_balanced_loss": 0.20038945792356924,
  "test_unweighted_genus_loss": 0.4373100415176275,
  "test_unweighted_health_loss": 0.4378860282654665,
  "test_unweighted_fish_loss": 1.1305469272501388e-08,
  "test_unweighted_human_artifacts_loss": 0.20578183637627837,
  "test_unweighted_substrate_loss": 0.5260443221397546,
  "test_unweighted_background_loss": 0.0998354976894619,
  "test_unweighted_biota_loss": 0.6386282006373667,
  "test_unweighted_aux_loss_sum": 1.470289869728137
}
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:369] - Optimization metrics summary:
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    tasks.genus.ungrouped.mIoU: 0.18834475154771596
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    tasks.genus.ungrouped.BIoU: 0.01579550759743983
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    tasks.genus.grouped.mIoU: 0.19777907931966288
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    tasks.genus.grouped.BIoU: 0.013585768511187648
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    tasks.health.ungrouped.mIoU: 0.19625195611605628
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    tasks.health.ungrouped.BIoU: 0.01528763137517262
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    tasks.health.grouped.mIoU: 0.15432595598393697
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    tasks.health.grouped.BIoU: 0.014188954190884052
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    tasks.fish.ungrouped.mIoU: 0.0032239531984134596
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    tasks.fish.ungrouped.BIoU: 0.0
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    tasks.human_artifacts.ungrouped.mIoU: 0.014265116919657985
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    tasks.human_artifacts.ungrouped.BIoU: 0.0035218195637992146
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    tasks.substrate.ungrouped.mIoU: 0.32343909796358544
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    tasks.substrate.ungrouped.BIoU: 0.033949123063802884
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    tasks.background.ungrouped.mIoU: 0.15126913537551623
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    tasks.background.ungrouped.BIoU: 0.020095228741868903
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    tasks.biota.ungrouped.mIoU: 0.017823063759064815
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    tasks.biota.ungrouped.BIoU: 0.0020526994600347306
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    global.mIoU: 0.08475513819647218
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    global.BIoU: 0.008225378539666952
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    global.classification_error: 0.7012753085214256
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    global.background_error: 0.18866779123033775
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    global.missed_error: 0.0
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    global.Boundary_F1: 0.016316058072420907
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    global.Boundary_Precision: 0.013766088558529734
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    global.Boundary_Recall: 0.020026946532500103
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    global.NLL: 14.44483470916748
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    global.Brier_Score: 1.1155517101287842
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    global.ECE: 0.3203518580493579
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    test_weighted_genus_loss: 0.0594511532167695
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    test_dwa_weight_genus: 0.13594737648963928
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    test_weighted_health_loss: 0.059546656830578436
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    test_dwa_weight_health: 0.13598665595054626
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    test_weighted_fish_loss: 2.10917737069473e-09
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    test_dwa_weight_fish: 0.18656256794929504
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    test_weighted_human_artifacts_loss: 0.027037290191605726
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    test_dwa_weight_human_artifacts: 0.1313881278038025
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    test_weighted_substrate_loss: 0.07152267786370096
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    test_dwa_weight_substrate: 0.13596321642398834
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    test_weighted_background_loss: 0.013573172810482698
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    test_dwa_weight_background: 0.13595537841320038
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    test_weighted_biota_loss: 0.08825631391426919
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    test_dwa_weight_biota: 0.13819670677185059
2025-09-29 09:41:31,100 - __main__ - DEBUG - [_log_key_metrics:371] -    test_total_loss: 0.31938726584217986
2025-09-29 09:41:31,101 - __main__ - DEBUG - [_log_key_metrics:371] -    test_consistency_loss: 0.0
2025-09-29 09:41:31,101 - __main__ - DEBUG - [_log_key_metrics:371] -    test_primary_balanced_loss: 0.11899781025642035
2025-09-29 09:41:31,101 - __main__ - DEBUG - [_log_key_metrics:371] -    test_aux_balanced_loss: 0.20038945792356924
2025-09-29 09:41:31,101 - __main__ - DEBUG - [_log_key_metrics:371] -    test_unweighted_genus_loss: 0.4373100415176275
2025-09-29 09:41:31,101 - __main__ - DEBUG - [_log_key_metrics:371] -    test_unweighted_health_loss: 0.4378860282654665
2025-09-29 09:41:31,101 - __main__ - DEBUG - [_log_key_metrics:371] -    test_unweighted_fish_loss: 1.1305469272501388e-08
2025-09-29 09:41:31,101 - __main__ - DEBUG - [_log_key_metrics:371] -    test_unweighted_human_artifacts_loss: 0.20578183637627837
2025-09-29 09:41:31,101 - __main__ - DEBUG - [_log_key_metrics:371] -    test_unweighted_substrate_loss: 0.5260443221397546
2025-09-29 09:41:31,101 - __main__ - DEBUG - [_log_key_metrics:371] -    test_unweighted_background_loss: 0.0998354976894619
2025-09-29 09:41:31,101 - __main__ - DEBUG - [_log_key_metrics:371] -    test_unweighted_biota_loss: 0.6386282006373667
2025-09-29 09:41:31,101 - __main__ - DEBUG - [_log_key_metrics:371] -    test_unweighted_aux_loss_sum: 1.470289869728137
2025-09-29 09:41:31,101 - __main__ - INFO - [run_single_experiment:301] - ✅ Evaluation completed successfully for mtl
2025-09-29 09:41:31,101 - __main__ - INFO - [run_single_experiment:311] - Total experiment duration: 2:48:25.786931
2025-09-29 09:41:31,101 - __main__ - INFO - [run_single_experiment:314] - 🎉 MTL experiment completed successfully!
2025-09-29 09:41:31,713 - __main__ - INFO - [run_comparison:433] - ✅ Experiment mtl completed successfully
2025-09-29 09:41:31,714 - __main__ - INFO - [run_comparison:408] - 
🚀 Running experiment 2/2: baseline
2025-09-29 09:41:31,714 - __main__ - DEBUG - [run_comparison:417] - Config path for baseline: /workspace/dbl_capstone/configs/baseline_comparisons/baseline_config.yaml
2025-09-29 09:41:31,714 - __main__ - DEBUG - [run_comparison:418] - Checkpoint path for baseline: None
2025-09-29 09:41:31,714 - __main__ - INFO - [run_single_experiment:211] - 
============================================================
2025-09-29 09:41:31,714 - __main__ - INFO - [run_single_experiment:212] - Starting BASELINE Experiment
2025-09-29 09:41:31,714 - __main__ - INFO - [run_single_experiment:213] - Config: /workspace/dbl_capstone/configs/baseline_comparisons/baseline_config.yaml
2025-09-29 09:41:31,714 - __main__ - INFO - [run_single_experiment:214] - Skip training: False
2025-09-29 09:41:31,714 - __main__ - INFO - [run_single_experiment:215] - Checkpoint path: Auto-detect best
2025-09-29 09:41:31,714 - __main__ - INFO - [run_single_experiment:216] - ============================================================
2025-09-29 09:41:31,714 - __main__ - DEBUG - [run_single_experiment:220] - Initializing ExperimentFactory with config: /workspace/dbl_capstone/configs/baseline_comparisons/baseline_config.yaml
2025-09-29 09:41:31,721 - __main__ - DEBUG - [run_single_experiment:222] - ExperimentFactory initialized successfully
2025-09-29 09:41:31,721 - __main__ - DEBUG - [run_single_experiment:237] - Logging experiment configuration details...
2025-09-29 09:41:31,721 - __main__ - INFO - [run_single_experiment:245] - Model type: SegFormerBaseline
2025-09-29 09:41:31,721 - __main__ - INFO - [run_single_experiment:246] - Primary tasks: []
2025-09-29 09:41:31,721 - __main__ - INFO - [run_single_experiment:247] - Auxiliary tasks: []
2025-09-29 09:41:31,721 - __main__ - INFO - [run_single_experiment:248] - Device policy: cuda
2025-09-29 09:41:31,721 - __main__ - INFO - [run_single_experiment:249] - Batch size per GPU: 8
2025-09-29 09:41:31,721 - __main__ - INFO - [run_single_experiment:250] - Learning rate: 6e-05
2025-09-29 09:41:31,721 - __main__ - INFO - [run_single_experiment:251] - Epochs: 50
2025-09-29 09:41:31,721 - __main__ - INFO - [run_single_experiment:252] - Output directory: /workspace/dbl_capstone/experiments/baseline_comparisons/coral_baseline_b2_run
2025-09-29 09:41:31,721 - __main__ - INFO - [run_single_experiment:258] - 🎯 Phase 1: Training & Validation for baseline
2025-09-29 09:41:31,721 - __main__ - DEBUG - [run_single_experiment:259] - Starting training phase...
2025-09-29 09:41:31,721 - __main__ - DEBUG - [run_single_experiment:262] - Training started at: 2025-09-29 09:41:31.721394
2025-09-29 11:53:31,588 - __main__ - INFO - [run_single_experiment:268] - Training duration: 2:11:59.867487
2025-09-29 11:53:31,589 - __main__ - INFO - [run_single_experiment:271] - ✅ Training completed successfully for baseline
2025-09-29 11:53:31,589 - __main__ - INFO - [run_single_experiment:283] - 🔍 Phase 2: Final Testing & Evaluation for baseline
2025-09-29 11:53:31,589 - __main__ - DEBUG - [run_single_experiment:284] - Starting evaluation phase...
2025-09-29 11:53:31,589 - __main__ - DEBUG - [run_single_experiment:287] - Evaluation started at: 2025-09-29 11:53:31.589386
2025-09-29 11:53:31,589 - __main__ - DEBUG - [run_single_experiment:288] - Using checkpoint: Auto-detect best
