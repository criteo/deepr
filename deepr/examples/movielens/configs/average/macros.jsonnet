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
        num_negatives: 1000,
        loss: "bpr"
    }
}
