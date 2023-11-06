local base = (import 'base.jsonnet');

base + {
    trainer+:{
        num_epochs: 300,
        optimizer:
            {
                type: 'SGD',
                lr: 0.1,
                weight_decay: 0.0005,
                momentum: 0.9,
                nesterov: 1,
            },
        scheduler:
            {
                type: 'CosineAnnealingLR',
                T_max: 300,
                eta_min: 0.0,
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