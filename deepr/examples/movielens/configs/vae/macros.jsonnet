local macros = import '../common/macros.jsonnet';

macros + {
    run+: {
        run_on_yarn: false
    },
    mlflow+: {
        use_mlflow: false
    },
    params+: {
        target_ratio: null,
        num_negatives: 100,
        loss: "multi_css"
    }
}
