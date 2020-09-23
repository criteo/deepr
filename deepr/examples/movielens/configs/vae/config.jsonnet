local start = import '../common/start.jsonnet';
local train = import '../common/train.jsonnet';
local evaluate = import '../common/evaluate.jsonnet';
local end = import '../common/end.jsonnet';

{
    type: "deepr.jobs.YarnLauncher",
    config: {
        type: "deepr.jobs.YarnLauncherConfig",
        hadoop_file_systems: [
            "viewfs://root",
            "viewfs://prod-am6",
        ],
        path_pex_cpu: "viewfs://root/user/g.genthial/envs/cpu/yarn-launcher-2020-09-22-17-39-04.pex",
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
                        type: "deepr.examples.movielens.layers.VAEModel",
                        vocab_size: "$params:vocab_size",
                        dims_encode: [600, 200],
                        dims_decode: [200, 600],
                        keep_prob: 0.5,
                    },
                    loss_fn: {
                        type: "deepr.examples.movielens.layers.VAELoss",
                        loss: "$params:loss",
                        vocab_size: "$params:vocab_size",
                        beta_start: 0,
                        beta_end: 0.2,
                        beta_steps: 40000,
                    }
                }
            },
            evaluate,
            end,
        ]
    }
}
