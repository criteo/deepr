{
    type: "deepr.jobs.Pipeline",
    jobs: [
        {
            type: "deepr.examples.movielens.jobs.Predict",
            path_saved_model: "$paths:path_saved_model",
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
        },
        {
            type: "deepr.examples.movielens.jobs.Evaluate",
            path_predictions: "$paths:path_predictions",
            path_embeddings: "$paths:path_embeddings",
            path_biases: "$paths:path_biases",
            k: [20, 50, 100],
            use_mlflow: "$mlflow:use_mlflow"
        }
    ]
}
