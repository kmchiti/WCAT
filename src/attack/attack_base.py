
from quantization import Quantizer, Quantized_Linear, Quantized_Conv2d, hamming_distance
from common import Registrable


def hamming_distance_model(model1, model2):
    H_dist = 0  # hamming distance counter
    for name, module1 in model1.named_modules():
        if isinstance(module1, Quantized_Conv2d) or isinstance(module1, Quantized_Linear):
            module2 = model2.get_submodule(name)

            weight1_int = module1.weight_quantize_module.linear_quantize(module1.weight)
            weight1_uint = module1.weight_quantize_module.int2bin(weight1_int)

            weight2_int = module1.weight_quantize_module.linear_quantize(module2.weight)
            weight2_uint = module1.weight_quantize_module.int2bin(weight2_int)
            H_dist += hamming_distance(weight1_uint, weight2_uint, module1.weight_quantize_module.N_bits)

    return H_dist


class Attack(Registrable):
    pass
