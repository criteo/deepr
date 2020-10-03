local macros = import '../common/macros.jsonnet';

macros + {
    run+: {
        run_on_yarn: true
    },
    mlflow+: {
        use_mlflow: true
    },
    params+: {
        take_ratio: null,
        target_ratio: null,
        num_negatives: null,
        loss: "multi",
        normalize_embeddings: true,
        dim: 600,
        train_embeddings: false,
        project: true,
        reduce_mode: "l2"
    }
}
