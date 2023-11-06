local base = (import 'base.jsonnet');

base + {
    attack+:{
        type: "rbf",
        noise_type: "random",
        protect_bit_index: "NoProtection",
        protect_bit_amount: null,
        module_list: null,
        attack_MC_sample: 50,
        },
}