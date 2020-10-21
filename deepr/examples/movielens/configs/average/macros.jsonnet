local macros = import '../common/macros.jsonnet';

macros + {
    run+: {
        run_on_yarn: false
    },
    mlflow+: {
        use_mlflow: false
    },
    params+: {
        dim: 600,
        take_ratio: null,
        target_ratio: 0.2,
        num_negatives: 1000,
        loss: "bpr",
        share_embeddings: true,
        average_with_bias: false,
        reduce_mode: "average"
    }
}
