{
    type: "deepr.jobs.Pipeline",
    jobs: [
        {
            type: "deepr.jobs.LogMetric",
            key: "SUCCESS",
            value: true,
            use_mlflow: "$mlflow:use_mlflow"
        },
        {
            type: "deepr.jobs.CleanupCheckpoints",
            path_model: "$paths:path_model"
        }
    ]
}
