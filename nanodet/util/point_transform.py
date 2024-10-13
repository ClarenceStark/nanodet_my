import torch

def distance2quadrilateral(points, distance):
    """Decode distance prediction to quadrilateral.

    Args:
        points (Tensor): Shape (n, 2), [cx, cy] for each point.
        distance (Tensor): Shape (n, 8), representing the distances to four corner points.

    Returns:
        Tensor: Decoded quadrilateral corner points, shape (n, 8).
    """
    p1_x = points[..., 0] - distance[..., 0]  # p1_x = cx - p1_x_offset
    p1_y = points[..., 1] - distance[..., 1]  # p1_y = cy - p1_y_offset
    p2_x = points[..., 0] - distance[..., 2]  # p2_x = cx - p2_x_offset
    p2_y = points[..., 1] - distance[..., 3]  # p2_y = cy - p2_y_offset
    p3_x = points[..., 0] + distance[..., 4]  # p3_x = cx + p3_x_offset
    p3_y = points[..., 1] + distance[..., 5]  # p3_y = cy + p3_y_offset
    p4_x = points[..., 0] + distance[..., 6]  # p4_x = cx + p4_x_offset
    p4_y = points[..., 1] + distance[..., 7]  # p4_y = cy + p4_y_offset

    return torch.stack([p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y], -1)

def warp_quads(quads, warp_matrix, img_width, img_height):
    """Apply inverse transformation to quadrilateral corners.

    Args:
        quads (numpy.ndarray): Shape (n, 8), quadrilateral corners.
        warp_matrix (numpy.ndarray): Inverse of the transformation matrix.
        img_width (int): Original image width.
        img_height (int): Original image height.

    Returns:
        numpy.ndarray: Transformed quadrilateral corners.
    """
    # Reshape to (n, 4, 2) for easier manipulation of each corner
    quads = quads.reshape(-1, 4, 2)

    # Apply the warp matrix to each corner
    for i in range(quads.shape[0]):
        for j in range(4):
            point = np.array([quads[i, j, 0], quads[i, j, 1], 1.0])  # homogeneous coordinates
            transformed_point = np.dot(warp_matrix, point)
            quads[i, j, 0] = transformed_point[0] / transformed_point[2]
            quads[i, j, 1] = transformed_point[1] / transformed_point[2]
    
    # Flatten back to (n, 8)
    quads = quads.reshape(-1, 8)

    # Clip the transformed coordinates to be within image bounds
    quads[:, 0::2] = np.clip(quads[:, 0::2], 0, img_width)
    quads[:, 1::2] = np.clip(quads[:, 1::2], 0, img_height)

    return quads

def quads2distance(points, quads):
    """
    计算四边形的角点到中心点的距离。
    
    Args:
        points (Tensor): 形状 (n, 2)，表示先验框的中心点 [cx, cy]。
        quads (Tensor): 形状 (n, 8)，表示四边形的四个角点 [x1, y1, x2, y2, x3, y3, x4, y4]。
    
    Returns:
        Tensor: 每个角点到中心点的距离，形状 (n, 8)。
    """
    x1, y1 = quads[..., 0], quads[..., 1]
    x2, y2 = quads[..., 2], quads[..., 3]
    x3, y3 = quads[..., 4], quads[..., 5]
    x4, y4 = quads[..., 6], quads[..., 7]
    
    # 计算每个角点到中心点的距离
    distances = torch.stack([
        points[..., 0] - x1, points[..., 1] - y1,
        points[..., 0] - x2, points[..., 1] - y2,
        points[..., 0] - x3, points[..., 1] - y3,
        points[..., 0] - x4, points[..., 1] - y4,
    ], dim=-1)
    
    return distances
