local start = import '../common/start.jsonnet';
local end = import '../common/end.jsonnet';

{
    type: "deepr.jobs.YarnLauncher",
    config: {
        type: "deepr.jobs.YarnLauncherConfig",
        path_pex_cpu: "viewfs://root/user/g.genthial/envs/cpu/yarn-launcher-2020-10-01-17-54-40.pex"
    },
    run_on_yarn: "$run:run_on_yarn",
    job: {
        type: "deepr.jobs.Pipeline",
        eval: null,
        jobs: [
            start,
            {
                type: "deepr.examples.movielens.jobs.SVD",
                path_csv: "$paths:path_train",
                path_embeddings: "$paths:path_embeddings_svd",
                vocab_size: "$params:vocab_size",
                dim: "$params:dim",
                min_count: 10,
                normalize: true,
            },
            {
                type: "deepr.examples.movielens.jobs.PredictSVD",
                path_embeddings: "$paths:path_embeddings_svd",
                path_predictions: "$paths:path_predictions",
                input_fn: {
                    type: "deepr.examples.movielens.readers.TestCSVReader",
                    path_csv_tr: "$paths:path_test_tr",
                    path_csv_te: "$paths:path_test_te",
                    vocab_size: "$params:vocab_size"
                },
                prepro_fn: {
                    type: "deepr.examples.movielens.prepros.CSVPrepro",
                    vocab_size: "$params:vocab_size",
                    batch_size: "$params:batch_size",
                    repeat_size: null,
                    prefetch_size: 1,
                    num_parallel_calls: 8,
                },
                normalize: true
            },
            {
                type: "deepr.examples.movielens.jobs.Evaluate",
                path_predictions: "$paths:path_predictions",
                path_embeddings: "$paths:path_embeddings_svd",
                k: [20, 50, 100],
                use_mlflow: "$mlflow:use_mlflow"
            },
            end,
        ]
    }
}
