import numpy as np

PATCH_SIZE = 512

def dynamic_axis_partition(points, scene_range, max_splits_per_axis=5, target_points=512):
    """
    Point cloud partitioning algorithm with strict constraints:
    - Maximum 5 splits per axis
    - Fixed 512 points per patch
    - Maximum 125 patches (5*5*5)
    """
    patches = []
    patch_coords = []

    # Z-axis partitioning - requires 512*4*4 points
    z_target = target_points * max_splits_per_axis * max_splits_per_axis  # 512*4*4
    z_coords = points[:, 2]
    z_splits, _ = calculate_splits(z_coords, scene_range[2], z_target, max_splits_per_axis)
    
    for z_idx in range(len(z_splits)-1):
        z_min, z_max = z_splits[z_idx], z_splits[z_idx+1]
        z_mask = (points[:, 2] >= z_min) & (points[:, 2] < z_max)
        z_layer = points[z_mask]
        if len(z_layer) == 0:
            continue

        # Y-axis partitioning - requires 512*4 points
        y_target = target_points * max_splits_per_axis  # 512*4
        y_coords = z_layer[:, 1]
        y_splits, _ = calculate_splits(y_coords, scene_range[1], y_target, max_splits_per_axis)

        for y_idx in range(len(y_splits)-1):
            y_min, y_max = y_splits[y_idx], y_splits[y_idx+1]
            y_mask = (z_layer[:, 1] >= y_min) & (z_layer[:, 1] < y_max)
            y_row = z_layer[y_mask]
            if len(y_row) == 0:
                continue

            # X-axis partitioning - requires 512 points
            x_coords = y_row[:, 0]
            x_splits, _ = calculate_splits(x_coords, scene_range[0], target_points, max_splits_per_axis)

            for x_idx in range(len(x_splits)-1):
                x_min, x_max = x_splits[x_idx], x_splits[x_idx+1]
                x_mask = (y_row[:, 0] >= x_min) & (y_row[:, 0] < x_max)
                patch = y_row[x_mask]
                
                # Maintain exactly 512 points
                processed_patch = adjust_points(patch, target_points)
                patch_coords.append((z_idx, y_idx, x_idx))
                patches.append(processed_patch)

    return patches, patch_coords


def calculate_splits(axis_coords, axis_range, target_points_per_split, max_splits):
    """Calculate split points based on point distribution"""
    sorted_coords = np.sort(axis_coords)
    total_points = len(sorted_coords)
    
    if total_points == 0:
        return [axis_range[0], axis_range[1]], 0
        
    # Split based on point distribution regardless of total points
    # Each partition should have at least target_points_per_split points
    required_splits = min(max_splits, max(1, total_points // target_points_per_split))
    
    # Determine split points based on cumulative point count ratio
    splits = [axis_range[0]]
    points_per_split = total_points / required_splits
    
    for i in range(1, required_splits):
        target_index = int(i * points_per_split)
        splits.append(sorted_coords[target_index])
    
    splits.append(axis_range[1])
    splits = list(np.unique(splits))
    
    return splits, len(splits)-1

def adjust_points(patch, target_points):
    """Strictly align the number of points to 512"""
    if len(patch) == 0:
        return np.zeros((target_points, patch.shape[1]))
    elif len(patch) > target_points:
        return fps_sampling(patch, target_points)
    else:
        repeat_times = (target_points // len(patch)) + 1
        return np.tile(patch, (repeat_times, 1))[:target_points]

def fps_sampling(points, n_samples, num_candidates=10):
    """Improved fast FPS sampling (fixed syntax errors)"""
    if len(points) <= n_samples:
        return points
    
    indices = [np.random.randint(len(points))]
    
    # Change loop variable to underscore (indicating the value is not used)
    for _ in range(1, n_samples):
        # Randomly select candidate points from remaining points
        remaining_mask = ~np.isin(np.arange(len(points)), indices)
        candidates = np.random.choice(np.where(remaining_mask)[0], num_candidates, replace=False)
        
        # Calculate minimum distance from each candidate to selected points
        dists = np.min(np.linalg.norm(
            points[candidates][:, None] - points[indices],  # Add new dimension for broadcasting
            axis=2  # Compute L2 norm after calculating differences for each axis
        ), axis=1)
        
        # Select candidate with maximum distance
        next_idx = candidates[np.argmax(dists)]
        indices.append(next_idx)
    
    return points[indices]
   