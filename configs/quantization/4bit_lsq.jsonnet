local base = (import 'base.jsonnet');

base + {
    quantizer+:{
        N_bits: 4,
        signed:1,
        p0: 0.0,
        type: "lsq",
        use_grad_scaled: 1,
        },
}