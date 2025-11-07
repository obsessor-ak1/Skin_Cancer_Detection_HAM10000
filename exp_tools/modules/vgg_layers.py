from torch import nn


def make_blocks(
    in_channels, out_channels, count=2, end_1x1=False, with_bn=False, init_weights=True
):
    layers = []
    for i in range(count):
        # Getting the params
        ksize = 3
        padding = 1
        stride = 1
        if i != count - 1:
            if end_1x1:
                ksize = 1
                padding = None
                stride = None

        if i != 0:
            in_channels = out_channels
        # Assembling the layers
        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=ksize,
            padding=padding,
            stride=stride,
        )
        layers.append(conv)
        if with_bn:
            layers.append(
                nn.BatchNorm2d(
                    out_channels,
                    eps=1e-05,
                    momentum=0.1,
                    affine=True,
                    track_running_stats=True,
                )
            )
        layers.append(nn.ReLU(inplace=True))
    # Adding the pooling layer
    layers.append(
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    block = nn.Sequential(*layers)
    if init_weights:
        initialize_weights(block)
    return block


def initialize_weights(module):
    """Initializes appropriate weights to different layers."""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


def get_classifier(dropout=0.5, num_classes=7):
    """Returns the classifier block of VGG architecture."""
    return nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(p=dropout),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(p=dropout),
        nn.Linear(4096, num_classes),
    )
