local path_dataset = "viewfs://root/user/deepr/dev/movielens/ml-20m/";

{
    run: {
        run_on_yarn: false,
        train_on_yarn: false
    },
    paths: {
        type: "deepr.examples.movielens.macros.Paths",
        path_ratings: path_dataset + "ratings.csv",
        path_unique_sid: path_dataset + "unique_sid.txt",
        path_unique_uid: path_dataset + "unique_uid.txt",
        path_train: path_dataset + "train.csv",
        path_eval_tr: path_dataset + "validation_tr.csv",
        path_eval_te: path_dataset + "validation_te.csv",
        path_test_tr: path_dataset + "test_tr.csv",
        path_test_te: path_dataset + "test_te.csv",
        path_root: null
    },
    mlflow: {
        type: "deepr.macros.MLFlowInit",
        use_mlflow: false,
        run_name: "$paths:run_name",
        tracking_uri: null,
        experiment_name: null,
        artifact_location: null,
    },
    params: {
        max_steps: 50000,
        batch_size: 512,
        vocab_size: {
            type: "deepr.vocab.size",
            path: "$paths:path_unique_sid"
        },
        target_ratio: null,
        save_checkpoints_steps: 230,
        take_ratio: null,
        num_negatives: null,
        loss: null,
        dim: 600,
    }
}
