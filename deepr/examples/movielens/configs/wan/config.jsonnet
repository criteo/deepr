{
    "type": "deepr.jobs.YarnLauncher",
    "config": {
        "type": "deepr.jobs.YarnLauncherConfig"
    },
    "run_on_yarn": "$run:run_on_yarn",
    "job": {
        "type": "deepr.jobs.Pipeline",
        "eval": null,
        "jobs": [
            {
                "type": "deepr.jobs.LogMetric",
                "key": "SUCCESS",
                "value": false,
                "use_mlflow": "$mlflow:use_mlflow"
            },
            {
                "type": "deepr.jobs.MLFlowSaveConfigs",
                "use_mlflow": "$mlflow:use_mlflow",
                "config": "@self",
                "macros": "@macros",
                "macros_eval": "@macros_eval",
                "formatter": {
                    "type": "deepr.jobs.MLFlowFormatter",
                    "include_keys": [
                        "path_model",
                        "path_dataset",
                        "eval_input_fn",
                        "exporters",
                        "initializer_fn",
                        "loss_fn",
                        "optimizer_fn",
                        "pred_fn",
                        "predict_input_fn",
                        "prepro_fn",
                        "train_input_fn",
                        "train_spec",
                        "eval_spec"
                    ]
                }
            },
            {
                "type": "deepr.examples.movielens.jobs.BuildRecords",
                "path_ratings": "$paths:path_ratings",
                "path_mapping": "$paths:path_mapping",
                "path_train": "$paths:path_record_train",
                "path_eval": "$paths:path_record_eval",
                "path_test": "$paths:path_record_test",
                "min_rating": 4,
                "min_length": 5,
                "num_negatives": 8,
                "target_ratio": 0.2,
                "size_test": 10000,
                "size_eval": 10000,
                "shuffle_timelines": true,
                "seed": 2020
            },
            {
                "type": "deepr.jobs.YarnTrainer",
                "config": {
                    "type": "deepr.jobs.YarnTrainerConfig"
                },
                "train_on_yarn": "$run:train_on_yarn",
                "trainer": {
                    "type": "deepr.jobs.Trainer",
                    "eval": null,
                    "path_model": "$paths:path_model",
                    "pred_fn": {
                        "type": "deepr.examples.movielens.layers.AverageModel",
                        "vocab_size": {
                            "type": "deepr.vocab.size",
                            "path": "$paths:path_mapping"
                        },
                        "dim": "$params:dim"
                    },
                    "loss_fn": {
                        "type": "deepr.examples.movielens.layers.BPRLoss",
                        "vocab_size": {
                            "type": "deepr.vocab.size",
                            "path": "$paths:path_mapping"
                        },
                        "dim": "$params:dim"
                    },
                    "optimizer_fn": {
                        "type": "deepr.optimizers.TensorflowOptimizer",
                        "optimizer": "Adam",
                        "learning_rate": 0.001
                    },
                    "train_input_fn": {
                        "type": "deepr.readers.TFRecordReader",
                        "path": "$paths:path_record_train",
                        "num_parallel_reads": 8,
                        "num_parallel_calls": 8,
                        "shuffle": true
                    },
                    "eval_input_fn": {
                        "type": "deepr.readers.TFRecordReader",
                        "path": "$paths:path_record_eval",
                        "num_parallel_reads": 8,
                        "num_parallel_calls": 8,
                        "shuffle": false
                    },
                    "prepro_fn": {
                        "type": "deepr.examples.movielens.prepros.RecordPrepro",
                        "min_input_size": 3,
                        "min_target_size": 3,
                        "max_input_size": 50,
                        "max_target_size": 50,
                        "buffer_size": 1024,
                        "batch_size": "$params:batch_size",
                        "repeat_size": null,
                        "prefetch_size": 1,
                        "num_parallel_calls": 8
                    },
                    "exporters": [
                        {
                            "type": "deepr.exporters.BestCheckpoint",
                            "metric": "triplet_precision",
                            "mode": "increase"
                        },
                        {
                            "type": "deepr.exporters.SaveVariables",
                            "path_variables": "$paths:path_variables",
                            "variable_names": [
                                "biases",
                                "embeddings"
                            ]
                        },
                        {
                            "type": "deepr.exporters.SavedModel",
                            "path_saved_model": "$paths:path_saved_model",
                            "fields": [
                                {
                                    "type": "deepr.utils.Field",
                                    "name": "inputPositives",
                                    "shape": [null],
                                    "dtype": "int64"
                                },
                                {
                                    "type": "deepr.utils.Field",
                                    "name": "inputMask",
                                    "shape": [null],
                                    "dtype": "bool"
                                }
                            ]
                        }
                    ],
                    "train_metrics": [
                        {
                            "type": "deepr.metrics.StepCounter",
                            "name": "num_steps"
                        },
                        {
                            "type": "deepr.metrics.DecayMean",
                            "decay": 0.98,
                            "tensors": [
                                "loss"
                            ]
                        }
                    ],
                    "eval_metrics": [
                        {
                            "type": "deepr.metrics.Mean",
                            "tensors": [
                                "loss",
                                "triplet_precision"
                            ]
                        }
                    ],
                    "final_metrics": [
                        {
                            "type": "deepr.metrics.Mean",
                            "tensors": [
                                "loss",
                                "triplet_precision"
                            ]
                        }
                    ],
                    "train_hooks": [
                        {
                            "type": "deepr.hooks.LoggingTensorHookFactory",
                            "functions": {
                                "memory_gb": {
                                    "type": "deepr.hooks.ResidentMemory",
                                    "unit": "gb"
                                },
                                "max_memory_gb": {
                                    "type": "deepr.hooks.MaxResidentMemory",
                                    "unit": "gb"
                                }
                            },
                            "name": "training",
                            "skip_after_step": "$params:max_steps",
                            "every_n_iter": 300,
                            "use_mlflow": "$mlflow:use_mlflow"
                        },
                        {
                            "type": "deepr.hooks.SummarySaverHookFactory",
                            "save_steps": 300
                        },
                        {
                            "type": "deepr.hooks.NumParamsHook",
                            "use_mlflow": "$mlflow:use_mlflow"
                        },
                        {
                            "type": "deepr.hooks.LogVariablesInitHook"
                        },
                        {
                            "type": "deepr.hooks.StepsPerSecHook",
                            "batch_size": "$params:batch_size",
                            "name": "training",
                            "use_mlflow": "$mlflow:use_mlflow",
                            "skip_after_step": 100000
                        },
                        {
                            "type": "deepr.hooks.EarlyStoppingHookFactory",
                            "metric": "triplet_precision",
                            "max_steps_without_improvement": 1000,
                            "min_steps": 5000,
                            "mode": "increase",
                            "run_every_steps": 300,
                            "final_step": 100000
                        }
                    ],
                    "eval_hooks": [
                        {
                            "type": "deepr.hooks.LoggingTensorHookFactory",
                            "name": "validation",
                            "at_end": true,
                            "use_mlflow": "$mlflow:use_mlflow"
                        }
                    ],
                    "final_hooks": [
                        {
                            "type": "deepr.hooks.LoggingTensorHookFactory",
                            "name": "final_validation",
                            "at_end": true,
                            "use_mlflow": "$mlflow:use_mlflow"
                        }
                    ],
                    "train_spec": {
                        "max_steps": "$params:max_steps"
                    },
                    "eval_spec": {
                        "steps": null,
                        "start_delay_secs": 30,
                        "throttle_secs": 30
                    },
                    "final_spec": {
                        "steps": null
                    },
                    "run_config": {
                        "type": "deepr.jobs.RunConfig",
                        "save_checkpoints_steps": 300,
                        "save_summary_steps": 300,
                        "keep_checkpoint_max": null,
                        "log_step_count_steps": 300
                    },
                    "config_proto": {
                        "type": "deepr.jobs.ConfigProto",
                        "inter_op_parallelism_threads": 8,
                        "intra_op_parallelism_threads": 8,
                        "log_device_placement": false,
                        "gpu_device_count": 0,
                        "cpu_device_count": 48
                    }
                }
            },
            {
                "type": "deepr.examples.movielens.jobs.Predict",
                "path_saved_model": "$paths:path_saved_model",
                "path_predictions": "$paths:path_predictions",
                "input_fn": {
                    "type": "deepr.readers.TFRecordReader",
                    "path": "$paths:path_record_test",
                    "num_parallel_reads": 8,
                    "num_parallel_calls": 8,
                    "shuffle": false
                },
                "prepro_fn": {
                    "type": "deepr.examples.movielens.prepros.RecordPrepro"
                }
            },
            {
                "type": "deepr.examples.movielens.jobs.Evaluate",
                "path_predictions": "$paths:path_predictions",
                "path_embeddings": "$paths:path_embeddings",
                "path_biases": "$paths:path_biases",
                "k": 20,
                "use_mlflow": "$mlflow:use_mlflow"
            },
            {
                "type": "deepr.examples.movielens.jobs.Evaluate",
                "path_predictions": "$paths:path_predictions",
                "path_embeddings": "$paths:path_embeddings",
                "path_biases": "$paths:path_biases",
                "k": 50,
                "use_mlflow": "$mlflow:use_mlflow"
            },
            {
                "type": "deepr.jobs.LogMetric",
                "key": "SUCCESS",
                "value": true,
                "use_mlflow": "$mlflow:use_mlflow"
            },
            {
                "type": "deepr.jobs.CleanupCheckpoints",
                "path_model": "$paths:path_model"
            }
        ]
    }
}
