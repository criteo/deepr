{
    type: "deepr.jobs.Pipeline",
    jobs: [
        {
            type: "deepr.jobs.LogMetric",
            key: "SUCCESS",
            value: false,
            use_mlflow: "$mlflow:use_mlflow"
        },
        {
            type: "deepr.jobs.MLFlowSaveConfigs",
            use_mlflow: "$mlflow:use_mlflow",
            config: "@self",
            macros: "@macros",
            macros_eval: "@macros_eval",
            formatter: {
                type: "deepr.jobs.MLFlowFormatter",
                include_keys: [
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
        }
    ]
}
