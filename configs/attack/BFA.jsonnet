local base = (import 'base.jsonnet');

base + {
    attack+:{
        type: "bfa",
        k_top: 100,
        attack_sample_size: 64,
        },
}