local macros = import '../common/macros.jsonnet';

macros + {
    run+: {
        run_on_yarn: false
    },
    mlflow+: {
        use_mlflow: false
    },
    params+: {
        max_steps: 50000,
        take_ratio: null,
        target_ratio: null,
        num_negatives: 10,
        loss: "multi_css",
        share_embeddings: true,
        average_with_bias: false,
        reduce_mode: "l2"
    }
}
