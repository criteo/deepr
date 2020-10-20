local macros = import '../common/macros.jsonnet';

macros + {
    run+: {
        run_on_yarn: false
    },
    mlflow+: {
        use_mlflow: false
    },
    params+: {
        take_ratio: null,
        target_ratio: 0.2,
        num_negatives: 1000,
        loss: "bpr",
        normalize_embeddings: true,
        dim: 600,
        train_embeddings: false,
        train_biases: true,
        project: true,
        reduce_mode: "average",
    }
}
