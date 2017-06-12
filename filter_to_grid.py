import tensorflow as tf


def filters_to_grid(filters, num_pad=1):
    """
    Normalize and arrange the filters into a grid of shape:
    [in_channels*height, out_channels*width]
    :param filters: convolutional filters of shape [height, width, in_channels, out_channels]
    :param num_pad: the number of zeros to pad in the both sides
    :return: filters in a grid
    """
    # min-max normalize the elements to be within [0,1]
    x_min = tf.reduce_min(filters)
    x_max = tf.reduce_max(filters)

    normalized_filters = (filters - x_min) / (x_max - x_min)

    # pad the normalized_filters with zeros in the both sides of height and width directions
    paddings = tf.constant([[num_pad, num_pad], [num_pad, num_pad], [0, 0], [0, 0]])
    padded_filters = tf.pad(normalized_filters, paddings, mode='CONSTANT')

    # to [in_channels, height, out_channels, width]
    transposed_filters = tf.transpose(padded_filters, [2, 0, 3, 1])

    # rearrange filters into a grid of shape:
    # [batch_size, in_channels*padded_height, out_channels*padded_width, image_channel]
    in_channels = filters.get_shape()[2].value
    out_channels = filters.get_shape()[3].value
    grid_height = in_channels * (filters.get_shape()[0].value + 2 * num_pad)
    grid_width = out_channels * (filters.get_shape()[1].value + 2 * num_pad)
    filters_grid = tf.reshape(transposed_filters,
                              [1, grid_height, grid_width, 1],
                              name="filters_grid")

    return filters_grid
