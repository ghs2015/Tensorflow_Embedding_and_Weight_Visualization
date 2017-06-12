import tensorflow as tf


def embeddings_to_grid(embeddings, grid_shape=None, num_pad=1):
    """
    Pad and arrange the embeddings into a grid.
    :param embeddings: embeddings of the shape [batch_size, height, width, depth]
    :param grid_shape: user defined grid shape for display
    :param num_pad: the number of zeros to pad in both sides
    :return: embeddings in a grid
    """
    # calculate the shape parameters to use
    embeddings_shape = tf.shape(embeddings)
    batch_size = embeddings_shape[0]
    padded_height = embeddings_shape[1] + 2 * num_pad
    padded_width = embeddings_shape[2] + 2 * num_pad
    num_embeddings = embeddings_shape[3]

    if grid_shape is None:
        num_row = num_embeddings
        num_column = 1
    else:
        num_row = grid_shape[0]
        num_column = grid_shape[1]

    grid_width = num_column * padded_width
    grid_height = num_row * padded_height

    # pad the embeddings with zeros in the both sides of height and width directions
    paddings = tf.constant([[0, 0], [num_pad, num_pad], [num_pad, num_pad], [0, 0]])
    padded_embeddings = tf.pad(embeddings, paddings, mode="CONSTANT")

    # transpose to [batch_size, depth, padded_height, padded_width]
    transposed_embeddings = tf.transpose(padded_embeddings, [0, 3, 1, 2])

    # reshape to : [batch_size*num_row, num_column, padded_height, padded_width]
    columns_extracted = tf.reshape(transposed_embeddings,
                                   [batch_size*num_row, num_column, padded_height, padded_width])

    # transpose to : [batch_size*num_row, padded_height, num_column, padded_width]
    column_rearranged = tf.transpose(columns_extracted, [0, 2, 1, 3])

    # reshape to : [batch_size*num_row, padded_height, num_column*padded_width, 1]
    column_done = tf.reshape(column_rearranged,
                             [batch_size*num_row, padded_height, grid_width, 1])

    # reshape to : [batch_size, num_row*padded_height, num_column*padded_width, 1]
    embeddings_grid = tf.reshape(column_done, [batch_size, grid_height, grid_width, 1])

    return embeddings_grid
