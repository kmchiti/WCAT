local base = (import 'base.jsonnet');

base + {
    model+:{
        type: "lenet5",
        num_classes: 10,
        pretrained: 0,
        device: "cuda",
        weight_quantize_module: {},
        },
}