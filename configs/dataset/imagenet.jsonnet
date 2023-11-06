local base = (import 'base.jsonnet');

base + {
    dataset+:{
        type: 'imagenet',
        batch_size: 256,
        _ILSVRC2012_img_path: '/network/datasets/imagenet',
        _imagenet_val_script: './src/imagenet_val.sh',
        _use_train: 0,
        },
}