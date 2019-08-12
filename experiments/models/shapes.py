def conv_2d_output(h_in, kernel_sz, padding=0, dilation=1, stride=1):
    """Computes the output of a Conv2D layer assuming square images."""
    return int((h_in + 2 * padding - dilation * (kernel_sz - 1) - 1) / stride + 1)


def maxpool_2d_output(h_in, kernel_sz, stride=None, padding=0, dilation=1):
    """Computes the output of a MaxPool2D layer assuming square images."""
    stride = stride or kernel_sz
    return int((h_in + 2 * padding - dilation * (kernel_sz - 1) - 1) / stride + 1)


def block_2d_output(h_ind: int, kernel_sz: int, maxpool_kernel_sz: int) -> int:
    conv_out = conv_2d_output(h_ind, kernel_sz)
    return maxpool_2d_output(conv_out, maxpool_kernel_sz)
