local start = import '../common/start.jsonnet';
local train = import '../common/train.jsonnet';
local evaluate = import '../common/evaluate.jsonnet';
local end = import '../common/end.jsonnet';

{
    type: "deepr.jobs.YarnLauncher",
    config: {
        type: "deepr.jobs.YarnLauncherConfig",
        path_pex_cpu: "viewfs://root/user/g.genthial/envs/cpu/yarn-launcher-2020-09-23-16-20-48.pex"
    },
    run_on_yarn: "$run:run_on_yarn",
    job: {
        type: "deepr.jobs.Pipeline",
        eval: null,
        jobs: [
            start,
            {
                type: "deepr.examples.movielens.jobs.SVD",
                path_csv: "$paths:path_train",
                path_embeddings: "$paths:path_embeddings_svd",
                vocab_size: "$params:vocab_size",
                dim: "$params:dim",
                min_count: 10,
                n_iter: 2,
            },
            {
                type: "deepr.examples.movielens.jobs.InitCheckpoint",
                path_embeddings: "$paths:path_embeddings_svd",
                path_init_ckpt: "$paths:path_init_ckpt",
                normalize: "$params:normalize_embeddings"
            },
            train + {
                trainer+: {
                    pred_fn: {
                        type: "deepr.examples.movielens.layers.AverageModel",
                        vocab_size: "$params:vocab_size",
                        dim: "$params:dim",
                        keep_prob: 0.5,
                        train_embeddings: "$params:train_embeddings"
                    },
                    loss_fn: {
                        type: "deepr.examples.movielens.layers.Loss",
                        loss: "$params:loss",
                        vocab_size: "$params:vocab_size",
                    },
                    initializer_fn: {
                        type: "deepr.initializers.CheckpointInitializer",
                        assignment_map: {"embeddings": "embeddings"},
                        path_init_ckpt: "$paths:path_init_ckpt"
                    },
                }
            },
            evaluate,
            end,
        ]
    }
}
