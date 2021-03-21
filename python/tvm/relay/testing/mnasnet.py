# github.com/pytorch/vision/blob/master/torchvision/models/mnasnet.py

from tvm import relay
from .init import create_workload
from . import layers

def inverted_residual(
    data,
    in_channel,
    out_channel,
    kernel_size,
    stride,
    expansion_factor,
    data_layout,
    kernel_layout,
    name="",
    lane=None,
):
    assert stride in [1, 2]
    assert kernel_size in [3, 5]
    mid_channel = in_channel * expansion_factor
    bn_axis = data_layout.index("C")
    iiinfo = {
        "table_size": (1, 256 * lane),
        "table_dtype": "uint8",
    } if lane else None

    conv1 = layers.conv2d(
        data=data,
        channels=mid_channel,
        kernel_size=1,
        data_layout=data_layout,
        kernel_layout=kernel_layout,
        name=name + "conv1",
        iiinfo=iiinfo,
    )
    bn1 = layers.batch_norm_infer(data=conv1, axis=bn_axis, name=name + "bn1")
    act1 = relay.nn.relu(data=bn1)
    conv2 = layers.conv2d(
        data=act1,
        channels=mid_channel,
        kernel_size=kernel_size,
        data_layout=data_layout,
        kernel_layout=kernel_layout,
        padding=kernel_size // 2,
        strides=stride,
        groups=mid_channel,
        name=name + "conv2",
        iiinfo=iiinfo,
    )
    bn2 = layers.batch_norm_infer(data=conv2, axis=bn_axis, name=name + "bn2")
    act2 = relay.nn.relu(data=bn2)
    conv3 = layers.conv2d(
        data=act2,
        channels=out_channel,
        kernel_size=1,
        data_layout=data_layout,
        kernel_layout=kernel_layout,
        name=name + "conv3",
    )
    bn3 = layers.batch_norm_infer(data=conv3, axis=bn_axis, name=name + "bn3")
    if in_channel == out_channel and stride == 1:
        return relay.add(bn3, data)
    else:
        return bn3
    

def stack(
    data,
    in_channel,
    out_channel,
    kernel_size,
    stride,
    exp_factor,
    repeats,
    data_layout,
    kernel_layout,
    name,
    lane,
):
    assert repeats >= 1
    body = inverted_residual(
        data=data,
        in_channel=in_channel,
        out_channel=out_channel,
        kernel_size=kernel_size,
        stride=stride,
        expansion_factor=exp_factor,
        kernel_layout=kernel_layout,
        data_layout=data_layout,
        name=name + "unit0_",
        lane=lane,
    )
    for i in range(repeats):
        body = inverted_residual(
            data=body,
            in_channel=in_channel,
            out_channel=out_channel,
            kernel_size=kernel_size,
            stride=1,
            expansion_factor=exp_factor,
            kernel_layout=kernel_layout,
            data_layout=data_layout,
            name=name + f"unit{i}_",
            lane=lane,
        )
    return body


def _round_to_multiple_of(val, divisor, round_up_bias=0.9):
    """ Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, unless the number is no more than 10% greater than the
    smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88. """
    assert 0.0 < round_up_bias < 1.0
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor


def _get_depths(alpha):
    """ Scales tensor depths as in reference MobileNet code, prefers rouding up
    rather than down. """
    depths = [32, 16, 24, 40, 80, 96, 192, 320]
    return [_round_to_multiple_of(depth * alpha, 8) for depth in depths]


def mnasnet(
    alpha,
    num_classes,
    data_shape,
    dropout,
    data_layout="NCHW",
    dtype="float32",
    lane=None,
):
    kernel_layout_n = "OIHW" if data_layout == "NCHW" else "HWIO"
    kernel_layout_i = "HWOI"
    kernel_layout = kernel_layout_n if not lane else kernel_layout_i

    iiinfo = {
        "table_size": (1, 256 * lane),
        "table_dtype": "uint8",
    } if lane else None

    depths = _get_depths(alpha)
    bn_axis = data_layout.index("C")

    data = relay.var("data", shape=data_shape, dtype=dtype)

    conv0 = layers.conv2d(
        data=data,
        channels=depths[0],
        kernel_size=3,
        padding=1,
        strides=2,
        data_layout=data_layout,
        kernel_layout=kernel_layout_n,
        name="conv0",
        iiinfo=iiinfo,
    )
    bn0 = layers.batch_norm_infer(
        data=conv0,
        axis=bn_axis,
        name="bn0",
    )
    act0 = relay.nn.relu(bn0)
    conv1 = layers.conv2d(
        data=act0,
        channels=depth[0],
        kernel_size=3,
        padding=1,
        groups=depths[0],
        data_layout=data_layout,
        kernel_layout=kernel_layout,
        name="conv1",
        iiinfo=iiinfo,
    )
    bn1 = layers.batch_norm_infer(
        data=conv1,
        axis=bn_axis,
        name="bn1",
    )
    act1 = relay.nn.relu(bn1)
    conv2 = layers.conv2d(
        data=act1,
        channels=depths[1],
        kernel_size=1,
        data_layout=data_layout,
        kernel_layout=kernel_layout,
        name="conv2",
        iiinfo=iiinfo,
    )
    body = layers.batch_norm_infer(
        data=conv2,
        axis=bn_axis,
        name="bn2",
    )

    kernel_list = [3, 5, 5, 3, 5, 3]
    stride_list = [2, 2, 2, 1 , 2, 1]
    expansion_list = [3, 3, 6, 6, 6, 6]
    repeat_list = [3, 3, 3, 2, 4, 1]

    for i in range(len(kernel_list)):
        body = stack(
            data=body,
            in_channel=depths[i+1],
            out_channel=depths[i+2],
            kernel_size=kernel_list[i],
            stride=stride_list[i],
            exp_factor=expansion_list[i],
            repeats=repeat_list[i],
            data_layout=data_layout,
            kernel_layout=kernel_layout,
            name=f"stack{i}_",
            lane=lane,
        )
    
    conv3 = layers.conv2d(
        data=body,
        channels=1280,
        kernel_size=1,
        data_layout=data_layout,
        kernel_layout=kernel_layout,
        name="conv3",
        iiinfo=iiinfo,
    )
    bn3 = layers.batch_norm_infer(
        data=conv3,
        axis=bn_axis,
        name="bn3",
    )
    act3 = relay.nn.relu(bn3)
    pool = relay.nn.global_avg_pool2d(data=act3, layout=data_layout)
    flat = relay.nn.batch_flatten(pool)
    dp = relay.nn.dropout(act3, dropout)
    net = layers.dense_add_bias(dp, units=num_classes)

    return net


def get_net(
    batch_size,
    alpha,
    num_classes,
    dropout,
    image_shape,
    layout="NCHW",
    dtype="float32",
    lane=None,
):
    data_shape = (batch_size,) + image_shape 
    return mnasnet(
        alpha,
        num_classes,
        data_shape,
        dropout,
        layout,
        dtype,
        lane,
    )

def get_workload(
    batch_size=1,
    alpha=1.0,
    num_classes=1000,
    dropout=0.2,
    image_shape=(3, 224, 224),
    layout="NCHW",
    dtype="float32",
    lane=None,
    **kwargs,
):
    net = get_net(
        batch_size,
        alpha,
        num_classes,
        dropout,
        image_shape,
        layout,
        dtype,
        lane,
    )
    return create_workload(net)