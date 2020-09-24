local macros = import '../common/macros.jsonnet';

macros + {
    run+: {
        run_on_yarn: true
    },
    mlflow+: {
        use_mlflow: true
    },
    params+: {
        target_ratio: 0.2,
        num_negatives: 100,
        loss: "bpr",
        dim: 600,
        train_embeddings: false,
        normalize_embeddings: true,
        project: false
    }
}
