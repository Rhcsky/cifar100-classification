import re

import torch.nn
import torch.nn.init

from classificator.bsconv.replacers import BSConvS_Replacer, BSConvU_Replacer
from classificator.bsconv.common import conv1x1_block, conv3x3_block, Classifier


###
# %% ResNet building blocks
###


class InitUnitSmall(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 preact=False):
        super().__init__()

        self.conv = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=1,
            use_bn=not preact,
            activation=None if preact else "relu")

    def forward(self, x):
        x = self.conv(x)
        return x


class StandardUnit(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride):
        super().__init__()
        self.use_projection = (in_channels != out_channels) or (stride != 1)

        self.conv1 = conv3x3_block(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.conv2 = conv3x3_block(in_channels=out_channels, out_channels=out_channels, stride=1, activation=None)
        if self.use_projection:
            self.projection = conv1x1_block(in_channels=in_channels, out_channels=out_channels, stride=stride,
                                            activation=None)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        if self.use_projection:
            residual = self.projection(x)
        else:
            residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + residual
        x = self.relu(x)
        return x


class ResNet(torch.nn.Module):
    def __init__(self,
                 channels,
                 num_classes,
                 init_unit_channels=16,
                 use_init_unit_large=False,
                 in_channels=3,
                 in_size=(32, 32),
                 use_data_batchnorm=True):
        super().__init__()
        self.use_data_batchnorm = use_data_batchnorm
        self.use_init_unit_large = use_init_unit_large
        self.in_size = in_size

        self.backbone = torch.nn.Sequential()

        # data batchnorm
        if self.use_data_batchnorm:
            self.backbone.add_module("data_bn", torch.nn.BatchNorm2d(num_features=in_channels))

        # init unit
        self.backbone.add_module("init_unit",
                                 InitUnitSmall(in_channels=in_channels, out_channels=init_unit_channels))

        # stages
        in_channels = init_unit_channels
        for stage_id, stage_channels in enumerate(channels):
            stage = torch.nn.Sequential()
            for unit_id, unit_channels in enumerate(stage_channels):
                stride = 2 if (unit_id == 0) and (stage_id != 0) else 1

                stage.add_module("unit{}".format(unit_id + 1),
                                 StandardUnit(in_channels=in_channels, out_channels=unit_channels, stride=stride))

                in_channels = unit_channels
            self.backbone.add_module("stage{}".format(stage_id + 1), stage)
        self.backbone.add_module("global_pool", torch.nn.AdaptiveAvgPool2d(output_size=1))

        # classifier
        self.classifier = Classifier(in_channels=in_channels, num_classes=num_classes)

        self.init_params()

    def init_params(self):
        # backbone
        for name, module in self.backbone.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
        # classifier
        self.classifier.init_params()

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


###
# %% model definitions
###


def build_resnet(num_classes,
                 units_per_stage,
                 width_multiplier=1.0):
    init_unit_channels = 16
    channels_in_stage = [16, 32, 64]
    channels = [[int(ch * width_multiplier)] * rep for (ch, rep) in zip(channels_in_stage, units_per_stage)]
    use_init_unit_large = False
    in_size = (32, 32)

    net = ResNet(channels=channels,
                 num_classes=num_classes,
                 init_unit_channels=init_unit_channels,
                 use_init_unit_large=use_init_unit_large,
                 in_size=in_size)
    return net


def get_wide_resnet(architecture='wrn28_3.0_bsconvs_p1d4', num_classes=100):
    units_per_stage = {
        "resnet20": [3, 3, 3],
        "resnet56": [9, 9, 9],

        "wrn16": [2, 2, 2],
        "wrn28": [4, 4, 4],
        "wrn40": [6, 6, 6],
    }

    # split architecture string into base part and BSConv part
    pattern = r"^(?P<base>resnet[0-9]+|wrn[0-9]+_[0-9]+\.[0-9]+)(_(?P<bsconv_variant>bsconvu|bsconvs_p[0-9]+d[0-9]+))?$"
    match = re.match(pattern, architecture)
    if match is None:
        raise ValueError("Model architecture '{}' is not supported".format(architecture))
    base = match.group("base")
    bsconv_variant = match.group("bsconv_variant")

    # determine the width_multiplier and the key for the units_per_stage lookup table
    if base.startswith('resnet'):
        key = base
        width_multiplier = 1.0
    else:
        split = base.split("_")
        key = split[0]
        width_multiplier = float(split[1])

    # check if model configuration is defined in the lookup table
    if key not in units_per_stage:
        raise ValueError("Model architecture '{}' is not supported".format(architecture))

    # build model
    model = build_resnet(
        num_classes=num_classes,
        units_per_stage=units_per_stage[key],
        width_multiplier=width_multiplier,
    )

    # apply BSConv
    if bsconv_variant is None:
        pass
    elif bsconv_variant == "bsconvu":
        replacer = BSConvU_Replacer(with_bn=False)
        model = replacer.apply(model)
    elif bsconv_variant.startswith("bsconvs_"):
        p_frac = [float(value) for value in bsconv_variant.split("_")[1][1:].split("d")]
        p = p_frac[0] / p_frac[1]
        replacer = BSConvS_Replacer(p=p, with_bn=True)
        model = replacer.apply(model)

    return model


if __name__ == '__main__':
    img = torch.rand((1, 3, 32, 32))

    model = get_wide_resnet()

    # from classificator.bsconv.profile import ModelProfiler
    #
    # profile = ModelProfiler(model)
    #
    # profile.print_results()
