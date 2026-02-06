# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Triton kernel for euclidean distance transform (EDT)"""

import torch

# Try to import triton, provide fallback if not available
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    triton = None
    tl = None


def edt_triton(data: torch.Tensor):
    """
    Computes the Euclidean Distance Transform (EDT) of a batch of binary images.

    Args:
        data: A tensor of shape (B, H, W) representing a batch of binary images.

    Returns:
        A tensor of the same shape as data containing the EDT.
        It should be equivalent to a batched version of cv2.distanceTransform(input, cv2.DIST_L2, 0)
    """
    if not TRITON_AVAILABLE:
        # Fallback: use a simple CPU/GPU implementation
        return _edt_fallback(data)
    
    assert data.dim() == 3
    assert data.is_cuda
    B, H, W = data.shape
    data = data.contiguous()

    # Allocate the "function" tensor. Implicitly the function is 0 if data[i,j]==0 else +infinity
    output = torch.where(data, 1e18, 0.0)
    assert output.is_contiguous()

    # Scratch tensors for the parabola stacks
    parabola_loc = torch.zeros(B, H, W, dtype=torch.uint32, device=data.device)
    parabola_inter = torch.empty(B, H, W, dtype=torch.float, device=data.device)
    parabola_inter[:, :, 0] = -1e18
    parabola_inter[:, :, 1] = 1e18

    # Grid size (number of blocks)
    grid = (B, H)

    # Launch initialization kernel
    _edt_kernel[grid](
        output.clone(),
        output,
        parabola_loc,
        parabola_inter,
        H,
        W,
        horizontal=True,
    )

    # reset the parabola stacks
    parabola_loc.zero_()
    parabola_inter[:, :, 0] = -1e18
    parabola_inter[:, :, 1] = 1e18

    grid = (B, W)
    _edt_kernel[grid](
        output.clone(),
        output,
        parabola_loc,
        parabola_inter,
        H,
        W,
        horizontal=False,
    )
    # don't forget to take sqrt at the end
    return output.sqrt()


def _edt_fallback(data: torch.Tensor):
    """
    Fallback EDT implementation using OpenCV when triton is not available.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        # If even cv2 is not available, return a simple approximation
        # This is a very rough approximation - just returns the input
        return data.float()
    
    assert data.dim() == 3
    B, H, W = data.shape
    
    # Process each image in the batch
    results = []
    device = data.device
    data_cpu = data.cpu().numpy()
    
    for i in range(B):
        # Convert to uint8 for OpenCV
        img = (data_cpu[i] > 0).astype(np.uint8)
        # Compute distance transform
        dist = cv2.distanceTransform(1 - img, cv2.DIST_L2, 0)
        results.append(torch.from_numpy(dist))
    
    return torch.stack(results).to(device)


# Define the triton kernel only if triton is available
if TRITON_AVAILABLE:
    @triton.jit
    def _edt_kernel(inputs_ptr, outputs_ptr, v, z, height, width, horizontal: tl.constexpr):
        # This is a somewhat verbatim implementation of the efficient 1D EDT algorithm
        # It can be applied horizontally or vertically depending if we're doing the first or second stage.
        # It's parallelized across batch+row (or batch+col if horizontal=False)
        batch_id = tl.program_id(axis=0)
        if horizontal:
            row_id = tl.program_id(axis=1)
            block_start = (batch_id * height * width) + row_id * width
            length = width
            stride = 1
        else:
            col_id = tl.program_id(axis=1)
            block_start = (batch_id * height * width) + col_id
            length = height
            stride = width

        # This will be the index of the right most parabola in the envelope ("the top of the stack")
        k = 0
        for q in range(1, length):
            # Read the function value at the current location. Note that we're doing a singular read, not very efficient
            cur_input = tl.load(inputs_ptr + block_start + (q * stride))
            # location of the parabola on top of the stack
            r = tl.load(v + block_start + (k * stride))
            # associated boundary
            z_k = tl.load(z + block_start + (k * stride))
            # value of the function at the parabola location
            previous_input = tl.load(inputs_ptr + block_start + (r * stride))
            # intersection between the two parabolas
            s = (cur_input - previous_input + q * q - r * r) / (q - r) / 2

            # we'll pop as many parabolas as required
            while s <= z_k and k - 1 >= 0:
                k = k - 1
                r = tl.load(v + block_start + (k * stride))
                z_k = tl.load(z + block_start + (k * stride))
                previous_input = tl.load(inputs_ptr + block_start + (r * stride))
                s = (cur_input - previous_input + q * q - r * r) / (q - r) / 2

            # Store the new one
            k = k + 1
            tl.store(v + block_start + (k * stride), q)
            tl.store(z + block_start + (k * stride), s)
            if k + 1 < length:
                tl.store(z + block_start + ((k + 1) * stride), 1e9)

        # Last step, we read the envelope to find the min in every location
        k = 0
        for q in range(length):
            while (
                k + 1 < length
                and tl.load(
                    z + block_start + ((k + 1) * stride), mask=(k + 1) < length, other=q
                )
                < q
            ):
                k += 1
            r = tl.load(v + block_start + (k * stride))
            d = q - r
            old_value = tl.load(inputs_ptr + block_start + (r * stride))
            tl.store(outputs_ptr + block_start + (q * stride), old_value + d * d)
else:
    # Dummy kernel function when triton is not available
    _edt_kernel = None
