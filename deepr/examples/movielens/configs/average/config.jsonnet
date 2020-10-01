local start = import '../common/start.jsonnet';
local train = import '../common/train.jsonnet';
local evaluate = import '../common/evaluate.jsonnet';
local end = import '../common/end.jsonnet';

{
    type: "deepr.jobs.YarnLauncher",
    config: {
        type: "deepr.jobs.YarnLauncherConfig"
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
                        keep_prob: 0.5,
                        share_embeddings: "$params:share_embeddings",
                        average_with_bias: "$params:average_with_bias",
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
