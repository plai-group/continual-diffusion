import torch


def one_level_wavelet_transform(image):
    stacked = torch.stack([image[::2, ::2, :], image[::2, 1::2, :], image[1::2, ::2, :], image[1::2, 1::2, :]], axis=1)
    # B x 4 x C x H/2 x W/2
    dwt_matrix = torch.tensor([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, -0.5, -0.5], [0.5, -0.5, 0.5, -0.5], [0.5, -0.5, -0.5, 0.5]])
    # 


def one_level_wavelet_transform(image):
    """
    Perform a one-level wavelet transform on an image using vectorized operations.
    Assumes the image is a numpy array of shape (height, width, channels).
    Returns LL, LH, HL, HH components.
    """
    rows_even = image[::2, :, :]
    rows_odd = image[1::2, :, :]
    cols_even_even = rows_even[:, ::2, :]
    cols_even_odd = rows_even[:, 1::2, :]
    cols_odd_even = rows_odd[:, ::2, :]
    cols_odd_odd = rows_odd[:, 1::2, :]
    ll = (cols_even_even + cols_even_odd + cols_odd_even + cols_odd_odd) // 4
    lh = (cols_even_even - cols_even_odd + cols_odd_even - cols_odd_odd) // 4
    hl = (cols_even_even + cols_even_odd - cols_odd_even - cols_odd_odd) // 4
    hh = (cols_even_even - cols_even_odd - cols_odd_even + cols_odd_odd) // 4
    return ll, lh, hl, hh


def two_level_wavelet_transform_refactored_v2(image):
    """
    Perform a two-level wavelet transform on an image by applying
    the one-level wavelet transform twice and editing the output tensor.
    """

    # First level decomposition
    ll1, lh1, hl1, hh1 = one_level_wavelet_transform(image)

    # Second level decomposition on LL component from first level
    ll2, lh2, hl2, hh2 = one_level_wavelet_transform(ll1)

    # Combine all components into a single tensor
    height, width, channels = ll2.shape
    transformed = np.zeros((height, width, channels * 16), dtype=np.int16)

    # Fill the components in the output tensor
    transformed[:, :, 0:channels] = ll2
    transformed[:, :, channels:2*channels] = lh2
    transformed[:, :, 2*channels:3*channels] = hl2
    transformed[:, :, 3*channels:4*channels] = hh2
    transformed[:, :, 4*channels:5*channels] = lh1[::2, ::2, :]
    transformed[:, :, 5*channels:6*channels] = lh1[::2, 1::2, :]
    transformed[:, :, 6*channels:7*channels] = lh1[1::2, ::2, :]

    # Fill remaining channels with other components similarly
    transformed[:, :, 7*channels:8*channels] = lh1[1::2, 1::2, :]
    transformed[:, :, 8*channels:9*channels] = hl1[::2, ::2, :]
    transformed[:, :, 9*channels:10*channels] = hl1[::2, 1::2, :]
    transformed[:, :, 10*channels:11*channels] = hl1[1::2, ::2, :]
    transformed[:, :, 11*channels:12*channels] = hl1[1::2, 1::2, :]
    transformed[:, :, 12*channels:13*channels] = hh1[::2, ::2, :]
    transformed[:, :, 13*channels:14*channels] = hh1[::2, 1::2, :]
    transformed[:, :, 14*channels:15*channels] = hh1[1::2, ::2, :]
    transformed[:, :, 15*channels:16*channels] = hh1[1::2, 1::2, :]

    return torch.from_numpy(transformed)


# Example usage
image_512_refactored_v2 = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
transformed_image_512_refactored_v2 = two_level_wavelet_transform_refactored_v2(image_512_refactored_v2)
print(transformed_image_512_refactored_v2.shape)


def inverse_wavelet_transform(transformed):
    """
    Perform the inverse of a two-level wavelet transform.
    Assumes transformed is a PyTorch tensor of shape (height, width, channels*16).
    Returns a numpy array representing the reconstructed image.
    """
    height, width, total_channels = transformed.shape
    channels = total_channels // 16

    # Extract components from the transformed tensor
    ll2 = transformed[:, :, 0:channels]
    lh2 = transformed[:, :, channels:2*channels]
    hl2 = transformed[:, :, 2*channels:3*channels]
    hh2 = transformed[:, :, 3*channels:4*channels]
    lh1 = np.zeros((height*2, width*2, channels), dtype=np.int16)
    lh1[::2, ::2, :] = transformed[:, :, 4*channels:5*channels]
    lh1[::2, 1::2, :] = transformed[:, :, 5*channels:6*channels]
    lh1[1::2, ::2, :] = transformed[:, :, 6*channels:7*channels]
    lh1[1::2, 1::2, :] = transformed[:, :, 7*channels:8*channels]
    # Similarly extract HL1 and HH1 components
    hl1 = np.zeros((height*2, width*2, channels), dtype=np.int16)
    hl1[::2, ::2, :] = transformed[:, :, 8*channels:9*channels]
    hl1[::2, 1::2, :] = transformed[:, :, 9*channels:10*channels]
    hl1[1::2, ::2, :] = transformed[:, :, 10*channels:11*channels]
    hl1[1::2, 1::2, :] = transformed[:, :, 11*channels:12*channels]
    hh1 = np.zeros((height*2, width*2, channels), dtype=np.int16)
    hh1[::2, ::2, :] = transformed[:, :, 12*channels:13*channels]
    hh1[::2, 1::2, :] = transformed[:, :, 13*channels:14*channels]
    hh1[1::2, ::2, :] = transformed[:, :, 14*channels:15*channels]
    hh1[1::2, 1::2, :] = transformed[:, :, 15*channels:16*channels]

    # Inverse transform for the second level
    ll1 = inverse_wavelet_level(ll2, lh2, hl2, hh2)

    # Inverse transform for the first level
    reconstructed_image = inverse_wavelet_level(ll1, lh1, hl1, hh1)

    return reconstructed_image

def inverse_one_level_wavelet_transform(ll, lh, hl, hh):
    """
    Inverse of the one-level wavelet transform for the Daub 5/3 wavelet.
    Reconstructs the original image from its wavelet components.
    """
    # Reconstruct the even and odd rows
    rows_even = ll + lh
    rows_odd = hl + hh

    # Initialize the reconstructed image
    height, width, channels = rows_even.shape
    reconstructed_image = np.zeros((height * 2, width * 2, channels), dtype=ll.dtype)

    # Interleave the rows and columns to reconstruct the original image
    reconstructed_image[::2, ::2, :] = rows_even
    reconstructed_image[1::2, ::2, :] = rows_odd
    reconstructed_image[::2, 1::2, :] = rows_even
    reconstructed_image[1::2, 1::2, :] = rows_odd

    return reconstructed_image

def inverse_wavelet_level(ll, lh, hl, hh):
    """
    Inverse of the one-level wavelet transform.
    Reconstructs the original image from its wavelet components.
    """
    # Initialize arrays to store the reconstructed rows
    rows_even = np.zeros_like(ll).repeat(2, axis=1)
    rows_odd = np.zeros_like(ll).repeat(2, axis=1)

    # Reconstruct the even and odd rows
    rows_even[:, ::2] = ll + lh
    rows_even[:, 1::2] = ll - lh
    rows_odd[:, ::2] = hl + hh
    rows_odd[:, 1::2] = hl - hh

    # Interleave the even and odd rows to reconstruct the original image
    reconstructed_image = np.zeros((ll.shape[0]*2, ll.shape[1]*2, ll.shape[2]), dtype=np.int16)
    reconstructed_image[::2, :, :] = rows_even
    reconstructed_image[1::2, :, :] = rows_odd

    return reconstructed_image


# Usage example
reconstructed_image = inverse_wavelet_transform(transformed_image_512_refactored_v2)
print(reconstructed_image - image_512_refactored_v2)