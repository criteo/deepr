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
        num_negatives: null,
        loss: "multi",
        share_embeddings: true,
        average_with_bias: false
    }
}
