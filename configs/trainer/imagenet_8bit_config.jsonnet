local base = (import 'base.jsonnet');

base + {
    trainer+:{
        num_epochs: 1,
        optimizer:
            {
                type: 'SGD',
                lr: 0.0001,
                weight_decay: 0.00005,
                momentum: 0.9,
                nesterov: 1,
            },
        scheduler:
            {
                type: 'ExponentialLR',
                gamma: 0.9589990,
            },
        q_range_weight_decay: 0.01,
        warmup_epochs: 0,
        warmup_next_method: null,
        max_mag_coef: null,
        grad_clip: null,
        _max_checkpoints: 5,
        _monitoring_range: false,
        },
}