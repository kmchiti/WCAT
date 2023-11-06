local base = (import 'base.jsonnet');

base + {
    model+:{
        type: "resnet20",
        num_classes: 10,
        pretrained: 1,
        device: "cuda",
        weight_quantize_module: {},
        },
}