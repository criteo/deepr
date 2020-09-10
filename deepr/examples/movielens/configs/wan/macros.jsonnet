local path_dataset = "../ml-20m/";

{
    "run": {
        "run_on_yarn": false,
        "train_on_yarn": false
    },
    "paths": {
        "type": "deepr.examples.movielens.macros.Paths",
        "path_ratings": path_dataset + "ratings.csv",
        "path_root": "model"
    },
    "mlflow": {
        "type": "deepr.macros.MLFlowInit",
        "use_mlflow": false,
        "run_name": "$paths:run_name",
        "tracking_uri": null,
        "experiment_name": null,
        "artifact_location": null
    },
    "params": {
        "dim": 600,
        "max_steps": 40000,
        "batch_size": 512
    }
}
