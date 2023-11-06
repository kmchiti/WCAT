local base = (import 'base.jsonnet');

base + {
    quantizer+:{
        N_bits: 4,
        signed:1,
        p0: 0.0,
        type: "wcat",
        use_grad_scaled: 1,
        init_method: "MSE",
        clip:
            {
                type: "ste",
            }
        },
}