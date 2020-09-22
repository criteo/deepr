local macros = import '../common/macros.jsonnet';

macros + {
    run+: {
        run_on_yarn: false
    },
    mlflow+: {
        use_mlflow: false
    },
    params+: {
        target_ratio: 0.2,
        num_negatives: 100,
        loss: "bpr"
    }
}
