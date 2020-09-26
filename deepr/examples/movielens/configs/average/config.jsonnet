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
            train + {
                trainer+: {
                    pred_fn: {
                        type: "deepr.examples.movielens.layers.AverageModel",
                        vocab_size: "$params:vocab_size",
                        dim: 600,
                        keep_prob: 0.5
                    },
                    loss_fn: {
                        type: "deepr.examples.movielens.layers.Loss",
                        loss: "$params:loss",
                        vocab_size: "$params:vocab_size",
                    }
                }
            },
            evaluate,
            end,
        ]
    }
}
