local base = (import 'base.jsonnet');

base + {
    model+:{
        type: "deit_small",
        num_classes: 1000,
        pretrained: true,
        device: "cuda",
        weight_quantize_module: {},
        },
}