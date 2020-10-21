{
    type: "deepr.jobs.YarnTrainer",
    config: {
        type: "deepr.jobs.YarnTrainerConfig"
    },
    train_on_yarn: "$run:train_on_yarn",
    trainer: {
        type: "deepr.jobs.Trainer",
        eval: null,
        path_model: "$paths:path_model",
        pred_fn: null,
        loss_fn: null,
        optimizer_fn: {
            type: "deepr.optimizers.TensorflowOptimizer",
            optimizer: "Adam",
            learning_rate: 0.001
        },
        train_input_fn: {
            type: "deepr.examples.movielens.readers.TrainCSVReader",
            path_csv: "$paths:path_train",
            vocab_size: "$params:vocab_size",
            target_ratio: "$params:target_ratio",
            take_ratio: "$params:take_ratio",
            seed: 42
        },
        eval_input_fn: {
            type: "deepr.examples.movielens.readers.TestCSVReader",
            path_csv_tr: "$paths:path_eval_tr",
            path_csv_te: "$paths:path_eval_te",
            vocab_size: "$params:vocab_size"
        },
        prepro_fn: {
            type: "deepr.examples.movielens.prepros.CSVPrepro",
            vocab_size: "$params:vocab_size",
            batch_size: "$params:batch_size",
            repeat_size: null,
            prefetch_size: 1,
            num_parallel_calls: 8,
            num_negatives: "$params:num_negatives",
        },
        train_metrics: [
            {
                type: "deepr.metrics.StepCounter",
                name: "num_steps"
            },
            {
                type: "deepr.metrics.DecayMean",
                decay: 0.98,
                pattern: "loss*"
            }
        ],
        eval_metrics: [
            {
                type: "deepr.metrics.Mean",
                pattern: "loss*",
            },
            {
                type: "deepr.examples.movielens.metrics.RecallAtK",
                name: "recall_at_20",
                k: 20,
                logits: "logits",
                inputs: "inputPositivesOneHot",
                targets: "targetPositives",
            },
            {
                type: "deepr.examples.movielens.metrics.RecallAtK",
                name: "recall_at_50",
                k: 50,
                logits: "logits",
                inputs: "inputPositivesOneHot",
                targets: "targetPositives",
            },
            {
                type: "deepr.examples.movielens.metrics.NDCGAtK",
                name: "ndcg_at_100",
                k: 100,
                logits: "logits",
                inputs: "inputPositivesOneHot",
                targets: "targetPositives",
            },
        ],
        train_hooks: [
            {
                type: "deepr.hooks.LoggingTensorHookFactory",
                functions: {
                    memory_gb: {
                        type: "deepr.hooks.ResidentMemory",
                        unit: "gb"
                    },
                    max_memory_gb: {
                        type: "deepr.hooks.MaxResidentMemory",
                        unit: "gb"
                    }
                },
                name: "training",
                skip_after_step: "$params:max_steps",
                every_n_iter: 100,
                use_mlflow: "$mlflow:use_mlflow"
            },
            {
                type: "deepr.hooks.SummarySaverHookFactory",
                save_steps: 100
            },
            {
                type: "deepr.hooks.NumParamsHook",
                use_mlflow: "$mlflow:use_mlflow"
            },
            {
                type: "deepr.hooks.LogVariablesInitHook"
            },
            {
                type: "deepr.hooks.StepsPerSecHook",
                batch_size: "$params:batch_size",
                name: "training",
                use_mlflow: "$mlflow:use_mlflow",
                skip_after_step: "$params:max_steps"
            }
        ],
        eval_hooks: [
            {
                type: "deepr.hooks.LoggingTensorHookFactory",
                name: "validation",
                at_end: true,
                use_mlflow: "$mlflow:use_mlflow"
            }
        ],
        final_hooks: [
            {
                type: "deepr.hooks.LoggingTensorHookFactory",
                name: "final_validation",
                at_end: true,
                use_mlflow: "$mlflow:use_mlflow"
            }
        ],
        train_spec: {
            max_steps: "$params:max_steps"
        },
        eval_spec: {
            steps: null,
            start_delay_secs: 30,
            throttle_secs: 30
        },
        final_spec: {
            steps: null
        },
        run_config: {
            type: "deepr.jobs.RunConfig",
            save_checkpoints_steps: "$params:save_checkpoints_steps",
            save_summary_steps: 100,
            keep_checkpoint_max: null,
            log_step_count_steps: 100
        },
        config_proto: {
            type: "deepr.jobs.ConfigProto",
            inter_op_parallelism_threads: 8,
            intra_op_parallelism_threads: 8,
            log_device_placement: false,
            gpu_device_count: 0,
            cpu_device_count: 48
        },
        exporters: [
            {
                type: "deepr.exporters.BestCheckpoint",
                metric: "ndcg_at_100",
                mode: "increase",
                use_mlflow: "$mlflow:use_mlflow"
            },
            {
                type: "deepr.exporters.SaveVariables",
                path_variables: "$paths:path_variables",
                variable_names: ["embeddings", "biases"]
            },
            {
                type: "deepr.exporters.SavedModel",
                path_saved_model: "$paths:path_saved_model",
                fields: [
                    {
                        type: "deepr.utils.Field",
                        name: "inputPositives",
                        shape: [null],
                        dtype: "int64"
                    },
                    {
                        type: "deepr.utils.Field",
                        name: "inputMask",
                        shape: [null],
                        dtype: "bool"
                    }
                ]
            }
        ]
    }
}
