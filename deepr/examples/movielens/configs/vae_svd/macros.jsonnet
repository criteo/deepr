local macros = import '../common/macros.jsonnet';

macros + {
    run+: {
        run_on_yarn: true
    },
    mlflow+: {
        use_mlflow: true
    },
    params+: {
        target_ratio: null,
        normalize_embeddings: true,
        share_embeddings: true,
        num_negatives: 1000,
        loss: "multi_css",
        train_embeddings: false,
        project: false,
    }
}
