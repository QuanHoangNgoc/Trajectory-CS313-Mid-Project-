def sub_sample_trajectory(trajectory, sample_n=30):
    """
        Sub sample a trajectory to a given number of points.
    """
    if not isinstance(trajectory, np.ndarray):
        raise TypeError("Trajectory must be of type np.ndarray")
    elif trajectory.shape[1] != 2:
        raise ValueError("Trajectory must be of shape (n, 2)")

    include = np.linspace(0, trajectory.shape[0]-1, sample_n, dtype=np.int32)
    return trajectory[include]


# Slope to rotation matrix
def slope_to_rotation_matrix(slope):
    """
        Convert slope to rotation matrix.
    """
    return np.array([[1, slope], [-slope, 1]])

def get_point_projection_on_line(point, line):
    """
        Get the projection of a point on a line.
    """

    # Get the slope of the line using the start and end points
    line_slope = (line[-1, 1] - line[0, 1]) / (line[-1, 0] - line[0, 0]) if line[-1, 0] != line[0, 0] else np.inf

    # In case the slope is infinite, we can directly get the projection
    if np.isinf(line_slope):
        return np.array([line[0,0], point[1]])
    
    # Convert the slope to a rotation matrix
    R = slope_to_rotation_matrix(line_slope)

    # Rotate the line and point
    rot_line = np.matmul(line, R.T)
    rot_point = np.matmul(point, R.T)

    # Get the projection
    proj = np.array([rot_point[0], rot_line[0,1]])

    # Undo the rotation for the projection
    R_inverse = np.linalg.inv(R)
    proj = np.matmul(proj, R_inverse.T)

    return proj


def partition2segments(partition):
    """
        Convert a partition to a list of segments.
    """

    if not isinstance(partition, np.ndarray):
        raise TypeError("partition must be of type np.ndarray")
    elif partition.shape[1] != 2:
        raise ValueError("partition must be of shape (n, 2)")
    
    segments = []
    for i in range(partition.shape[0]-1):
        segments.append(np.array([[partition[i, 0], partition[i, 1]], [partition[i+1, 0], partition[i+1, 1]]]))

    return segments


################# EQUATIONS #################
# Perpendicular Distance
def d_perpendicular(l1, l2):
    """
        Calculate the perpendicular distance between two lines.
    """
    # Find the shorter line and assign that as l_shorter
    l_shorter = l_longer = None
    l1_len, l2_len = d_euclidean(l1[0], l1[-1]), d_euclidean(l2[0], l2[-1])
    if l1_len < l2_len:
        l_shorter = l1
        l_longer = l2
    else:
        l_shorter = l2
        l_longer = l1

    ps = get_point_projection_on_line(l_shorter[0], l_longer)
    pe = get_point_projection_on_line(l_shorter[-1], l_longer)

    lehmer_1 = d_euclidean(l_shorter[0], ps)
    lehmer_2 = d_euclidean(l_shorter[-1], pe)

    if lehmer_1 == 0 and lehmer_2 == 0:
        return 0
    return (lehmer_1**2 + lehmer_2**2) / (lehmer_1 + lehmer_2)#, ps, pe, l_shorter[0], l_shorter[-1]
    
# Parallel Distance
def d_parallel(l1, l2):
    """
        Calculate the parallel distance between two lines.
    """
    # Find the shorter line and assign that as l_shorter
    l_shorter = l_longer = None
    l1_len, l2_len = d_euclidean(l1[0], l1[-1]), d_euclidean(l2[0], l2[-1])
    if l1_len < l2_len:
        l_shorter = l1
        l_longer = l2
    else:
        l_shorter = l2
        l_longer = l1

    ps = get_point_projection_on_line(l_shorter[0], l_longer)
    pe = get_point_projection_on_line(l_shorter[-1], l_longer)

    parallel_1 = min(d_euclidean(l_longer[0], ps), d_euclidean(l_longer[-1], ps))
    parallel_2 = min(d_euclidean(l_longer[0], pe), d_euclidean(l_longer[-1], pe))

    return min(parallel_1, parallel_2)

# Angular Distance
def d_angular(l1, l2, directional=True):
    """
        Calculate the angular distance between two lines.
    """

    # Find the shorter line and assign that as l_shorter
    l_shorter = l_longer = None
    l1_len, l2_len = d_euclidean(l1[0], l1[-1]), d_euclidean(l2[0], l2[-1])
    if l1_len < l2_len:
        l_shorter = l1
        l_longer = l2
    else:
        l_shorter = l2
        l_longer = l1

    # Get the minimum intersecting angle between both lines
    shorter_slope = (l_shorter[-1,1] - l_shorter[0,1]) / (l_shorter[-1,0] - l_shorter[0,0]) if l_shorter[-1,0] - l_shorter[0,0] != 0 else np.inf
    longer_slope = (l_longer[-1,1] - l_longer[0,1]) / (l_longer[-1,0] - l_longer[0,0]) if l_longer[-1,0] - l_longer[0,0] != 0 else np.inf

    # The case of a vertical line
    theta = None
    if np.isinf(shorter_slope):
        # Get the angle of the longer line with the x-axis and subtract it from 90 degrees
        tan_theta0 = longer_slope
        tan_theta1 = tan_theta0 * -1
        theta0 = np.abs(np.arctan(tan_theta0))
        theta1 = np.abs(np.arctan(tan_theta1))
        theta = min(theta0, theta1)
    elif np.isinf(longer_slope):
        # Get the angle of the shorter line with the x-axis and subtract it from 90 degrees
        tan_theta0 = shorter_slope
        tan_theta1 = tan_theta0 * -1
        theta0 = np.abs(np.arctan(tan_theta0))
        theta1 = np.abs(np.arctan(tan_theta1))
        theta = min(theta0, theta1)
    else:
        tan_theta0 = (shorter_slope - longer_slope) / (1 + shorter_slope * longer_slope)
        tan_theta1 = tan_theta0 * -1

        theta0 = np.abs(np.arctan(tan_theta0))
        theta1 = np.abs(np.arctan(tan_theta1))

        theta = min(theta0, theta1)

    if directional:
        return np.sin(theta) * d_euclidean(l_longer[0], l_longer[-1])

    if 0 <= theta < (90 * np.pi / 180):
        return np.sin(theta) * d_euclidean(l_longer[0], l_longer[-1])
    elif (90 * np.pi / 180) <= theta <= np.pi:
        return np.sin(theta)
    else:
        raise ValueError("Theta is not in the range of 0 to 180 degrees.")

# Total Trajectory Distance
def distance(l1, l2, directional=True, w_perpendicular=1, w_parallel=1, w_angular=1):
    """
        Get the total trajectory distance using all three distance formulas.
    """

    perpendicular_distance = d_perpendicular(l1, l2)
    parallel_distance = d_parallel(l1, l2)
    angular_distance = d_angular(l1, l2, directional=directional)

    return (w_perpendicular * perpendicular_distance) + (w_parallel * parallel_distance) + (w_angular * angular_distance)


# Minimum Description Length
def minimum_desription_length(start_idx, curr_idx, trajectory, w_angular=1, w_perpendicular=1, par=True, directional=True):
    """
        Calculate the minimum description length.
    """
    LH = LDH = 0
    for i in range(start_idx, curr_idx-1):
        ed = d_euclidean(trajectory[i], trajectory[i+1])
        LH += max(0, np.log2(ed, where=ed>0))
        if par:
            for j in range(start_idx, i-1):
                # print()
                # print(np.array([trajectory[start_idx], trajectory[i]]))
                # print(np.array([trajectory[j], trajectory[j+1]]))
                LDH += w_perpendicular * d_perpendicular(np.array([trajectory[start_idx], trajectory[i]]), np.array([trajectory[j], trajectory[j+1]]))
                LDH += w_angular * d_angular(np.array([trajectory[start_idx], trajectory[i]]), np.array([trajectory[j], trajectory[j+1]]), directional=directional)
    if par:
        return LH + LDH
    return LH


#############################################
def partition(trajectory, directional=True, progress_bar=False, w_perpendicular=1, w_angular=1):
    """
        Partition a trajectory into segments.
    """

    # Ensure that the trajectory is a numpy array of shape (n, 2)
    if not isinstance(trajectory, np.ndarray):
        raise TypeError("Trajectory must be a numpy array")
    elif trajectory.shape[1] != 2:
        raise ValueError("Trajectory must be a numpy array of shape (n, 2)")

    # Initialize the characteristic points, add the first point as a characteristic point
    cp_indices = []
    cp_indices.append(0)

    traj_len = trajectory.shape[0]
    start_idx = 0
    
    length = 1
    while start_idx + length < traj_len:
        if progress_bar:
            print(f'\r{round(((start_idx + length) / traj_len) * 100, 2)}%', end='')
        # print(f'Current Index: {start_idx + length}, Trajectory Length: {traj_len}')
        curr_idx = start_idx + length
        # print(start_idx, curr_idx)
        # print(f"Current Index: {curr_idx}, Current point: {trajectory[curr_idx]}")
        cost_par = minimum_desription_length(start_idx, curr_idx, trajectory, w_angular=w_angular, w_perpendicular=w_perpendicular, directional=directional)
        cost_nopar = minimum_desription_length(start_idx, curr_idx, trajectory, par=False, directional=directional)
        # print(f'Cost with partition: {cost_par}, Cost without partition: {cost_nopar}')
        if cost_par > cost_nopar:
            # print(f"Added characteristic point: {trajectory[curr_idx-1]} with index {curr_idx-1}")
            cp_indices.append(curr_idx-1)
            start_idx = curr_idx-1
            length = 1
        else:
            length += 1
    
    # Add last point to characteristic points
    cp_indices.append(len(trajectory) - 1)
    # print(cp_indices)
    
    return np.array([trajectory[i] for i in cp_indices])



