local path_dataset = "viewfs://root/user/g.genthial/movielens/ml-20m/";

{
    run: {
        run_on_yarn: false,
        train_on_yarn: false
    },
    paths: {
        type: "deepr.examples.movielens.macros.Paths",
        path_train: path_dataset + "train.csv",
        path_eval_tr: path_dataset + "validation_tr.csv",
        path_eval_te: path_dataset + "validation_te.csv",
        path_test_tr: path_dataset + "test_tr.csv",
        path_test_te: path_dataset + "test_te.csv",
    },
    mlflow: {
        type: "deepr.macros.MLFlowInit",
        use_mlflow: false,
        run_name: "$paths:run_name",
        tracking_uri: "https://mlflow.par.prod.crto.in",
        experiment_name: "deepr-nn",
        artifact_location: "viewfs://root/user/deepr/dev/mlflow_artifacts"
    },
    params: {
        max_steps: 50000,
        batch_size: 512,
        vocab_size: {
            type: "deepr.vocab.size",
            path: path_dataset + "unique_sid.txt"
        },
        target_ratio: null,
        num_negatives: null,
        loss: null,
    }
}
