local base = (import 'base.jsonnet');

base + {
    dataset+:{
        type: 'cifar10',
        batch_size: 256,
        valid_size: 0.1,
        },
}