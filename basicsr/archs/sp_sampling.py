
import torch


def SP_global_sampling(
    x,
    sp_map,
    output=0,
    tp='bhwc',
    return_inverse=False,
    return_valid_mask=False
):
    """
    Superpixel-based global sampling using mean pooling.

    Args:
        x:
            Feature map.
            [B,H,W,C] if tp='bhwc'
            [B,C,H,W] if tp='bchw'

        sp_map:
            Superpixel labels, shape [B,H,W].
            Labels must be non-negative integers.

        output:
            0: Return superpixel features (Q)
            1: Return superpixel features (K/V)
            2: Return (X_SP, X_SP) for graph construction

        tp:
            Feature map format: 'bhwc' or 'bchw'.

        return_inverse:
            If True, return compact SP index map [B,H,W].

        return_valid_mask:
            If True, return valid node mask [B,S_max].
            True = real node
            False = padding node

    Returns:
        output=0 or 1:
            X_SP: [B,S_max,C]

        output=2:
            (X_SP, X_SP)

        Optional outputs:
            inverse_map: [B,H,W]
            valid_mask: [B,S_max]
    """

    # --------------------------------------------------
    # 1. Convert feature map to BHWC
    # --------------------------------------------------

    if tp == 'bchw':
        x = x.permute(0, 2, 3, 1)

    elif tp != 'bhwc':
        raise ValueError(
            f"Unsupported tensor format: {tp}"
        )

    B, H, W, C = x.shape

    # --------------------------------------------------
    # 2. Validate inputs
    # --------------------------------------------------

    if sp_map.shape != (B, H, W):
        raise ValueError(
            f"Expected sp_map shape {(B,H,W)}, "
            f"got {tuple(sp_map.shape)}"
        )

    if sp_map.is_floating_point():
        raise TypeError(
            "sp_map must contain integer Superpixel IDs."
        )

    sp_map = sp_map.to(
        device=x.device,
        dtype=torch.long
    )

    if torch.any(sp_map < 0):
        raise ValueError(
            "Superpixel IDs must be non-negative."
        )

    if H == 0 or W == 0:
        raise ValueError("Feature map cannot have empty spatial dimensions.")

    if output not in (0, 1, 2):
        raise ValueError(
            f"Unsupported output: {output}"
        )

    # --------------------------------------------------
    # 3. Compact Superpixel IDs
    # --------------------------------------------------

    inverse_map = torch.empty_like(sp_map)

    sp_counts = []

    max_sp = 0

    for b in range(B):

        unique_ids, inverse = torch.unique(
            sp_map[b].reshape(-1),
            sorted=True,
            return_inverse=True
        )

        num_sp = unique_ids.numel()

        inverse_map[b] = inverse.reshape(H, W)

        sp_counts.append(num_sp)

        max_sp = max(max_sp, num_sp)

    # --------------------------------------------------
    # 4. Create valid node mask
    # --------------------------------------------------

    valid_mask = torch.zeros(
        B,
        max_sp,
        dtype=torch.bool,
        device=x.device
    )

    for b in range(B):

        valid_mask[b, :sp_counts[b]] = True

    # --------------------------------------------------
    # 5. Superpixel mean pooling
    # --------------------------------------------------

    X_SP = x.new_zeros(
        B,
        max_sp,
        C
    )

    for b in range(B):

        labels = inverse_map[b].reshape(-1)

        features = x[b].reshape(
            H * W,
            C
        )

        # Sum pixel features belonging to each SP
        sums = x.new_zeros(
            max_sp,
            C
        )

        sums.index_add_(
            0,
            labels,
            features
        )

        # Count pixels in each SP
        counts = torch.bincount(
            labels,
            minlength=max_sp
        ).to(dtype=x.dtype)

        # Mean pooling
        X_SP[b] = (
            sums / counts.clamp_min(1).unsqueeze(-1)
        )

    # --------------------------------------------------
    # 6. Prepare output
    # --------------------------------------------------

    if output == 2:
        results = [X_SP, X_SP]

    else:
        results = [X_SP]

    if return_inverse:
        results.append(inverse_map)

    if return_valid_mask:
        results.append(valid_mask)

    if len(results) == 1:
        return results[0]

    return tuple(results)


def SP_broadcast(x_sp, inverse_map):

    """
    Broadcast Superpixel features back to pixel space.

    Args:
        x_sp:
            Superpixel features [B,S_max,C]

        inverse_map:
            Compact Superpixel index map [B,H,W]

    Returns:
        x_out:
            Pixel-level features [B,H,W,C]
    """

    B, H, W = inverse_map.shape

    if x_sp.shape[0] != B:
        raise ValueError(
            "Batch size mismatch between x_sp and inverse_map."
        )

    C = x_sp.shape[-1]

    index = inverse_map.reshape(
        B,
        H * W
    )

    if index.numel() > 0 and index.max() >= x_sp.shape[1]:
        raise ValueError(
            "inverse_map contains indices outside x_sp."
        )

    x_out = torch.gather(
        x_sp,
        dim=1,
        index=index.unsqueeze(-1).expand(-1, -1, C)
    )

    return x_out.reshape(
        B,
        H,
        W,
        C
    )
