local macros = import '../common/macros.jsonnet';

macros + {
    paths+: {
        path_root: "model"
    },
    run+: {
        run_on_yarn: false
    },
    mlflow+: {
        use_mlflow: false
    },
    params+: {
        dim: 600
    }
}
