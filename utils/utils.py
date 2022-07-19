import jittor as jt
import jittor.nn as nn


def preprocess_input(opt, data):
    data['label'] = data['label'].long()
    label_map = data['label']
    bs, _, h, w = label_map.size()
    nc = opt.semantic_nc
    input_label = jt.float32(jt.zeros((bs, nc, h, w)))  # bs,30,256,256
    src = jt.float32(jt.ones((bs, nc, h, w)))
    input_semantics = input_label.scatter_(1, label_map, src)
    return data['image'].float32(), input_semantics


def start_grad(model):
    for param in model.parameters():
        if 'running_mean' in param.name() or 'running_var' in param.name(): continue
        if 'weight_u' in param.name() or 'weight_v' in param.name(): continue
        param.start_grad()


def stop_grad(model):
    for param in model.parameters():
        param.stop_grad()


def generate_labelmix(label, fake_image, real_image):
    target_map = jt.argmax(label, dim=1, keepdims=True)[0]
    all_classes = jt.unique(target_map)
    for c in all_classes:
        target_map[target_map == c] = jt.randint(0, 2, (1,))
    target_map = target_map.float()
    mixed_image = target_map * real_image + (1 - target_map) * fake_image
    return mixed_image, target_map


def print_parameters(network):
    param_count = 0
    for name, module in network.named_modules():
        if (isinstance(module, nn.Conv2d)
                or isinstance(module, nn.Linear)
                or isinstance(module, nn.Embedding)):
            param_count += sum([p.numel() for p in module.parameters()])
    print('Created', network.__class__.__name__, "with %d parameters" % param_count)
