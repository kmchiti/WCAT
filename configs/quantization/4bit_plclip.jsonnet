local base = (import 'base.jsonnet');

base + {
    quantizer+: {
        N_bits: 4,
        signed: 1,
        p0: 0.0,
        type: 'plclip',
        plclip: 0.1,
    },
}