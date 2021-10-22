"""
Math auxilary files
"""
import numpy as np

# using this math: https://en.wikipedia.org/wiki/Rotation_matrix
def rotation_matrix(yaw):
    """ Returns rotation matrix """
    val_ty = yaw

    val_ry = np.array( \
        [[np.cos(val_ty), 0, np.sin(val_ty)], [0, 1, 0], [-np.sin(val_ty), 0, np.cos(val_ty)]])

    return val_ry.reshape([3, 3])
    # return np.dot(np.dot(Rz,Ry), Rx)

# option to rotate and shift (for label info)
def create_corners(dimension, location=None, val_r=None):
    """ Creates corners """
    val_dx = dimension[2] / 2
    val_dy = dimension[0] / 2
    val_dz = dimension[1] / 2

    x_corners = []
    y_corners = []
    z_corners = []

    for i in [1, -1]:
        for j in [1, -1]:
            for k in [1, -1]:
                x_corners.append(val_dx*i)
                y_corners.append(val_dy*j)
                z_corners.append(val_dz*k)

    corners = [x_corners, y_corners, z_corners]

    # rotate if R is passed in
    if val_r is not None:
        corners = np.dot(val_r, corners)

    # shift if location is passed in
    if location is not None:
        for i, loc in enumerate(location):
            corners[i, :] = corners[i, :] + loc

    final_corners = []
    for i in range(8):
        final_corners.append([corners[0][i], corners[1][i], corners[2][i]])


    return final_corners

# this is based on the paper. Math!
# calib is a 3x4 matrix, box_2d is [(xmin, ymin), (xmax, ymax)]
# Math help: http://ywpkwon.github.io/pdf/bbox3d-study.pdf
def calc_location(dimension, proj_matrix, box_2d, alpha, theta_ray):
    """ Calculates location """
    #global orientation
    orient = alpha + theta_ray
    val_r = rotation_matrix(orient)

    # format 2d corners
    xmin = box_2d[0][0]
    ymin = box_2d[0][1]
    xmax = box_2d[1][0]
    ymax = box_2d[1][1]

    # left top right bottom
    box_corners = [xmin, ymin, xmax, ymax]

    # get the point constraints
    constraints = []

    left_constraints = []
    right_constraints = []
    top_constraints = []
    bottom_constraints = []

    # using a different coord system
    val_dx = dimension[2] / 2
    val_dy = dimension[0] / 2
    val_dz = dimension[1] / 2

    # below is very much based on trial and error

    # based on the relative angle, a different configuration occurs
    # negative is back of car, positive is front
    left_mult = 1
    right_mult = -1

    # about straight on but opposite way
    if np.deg2rad(88) < alpha < np.deg2rad(92):
        left_mult = 1
        right_mult = 1
    # about straight on and same way
    elif np.deg2rad(-92) < alpha < np.deg2rad(-88):
        left_mult = -1
        right_mult = -1
    # this works but doesnt make much sense
    elif -np.deg2rad(90) < alpha < np.deg2rad(90):
        left_mult = -1
        right_mult = 1

    # if the car is facing the oppositeway, switch left and right
    switch_mult = -1
    if alpha > 0:
        switch_mult = 1

    # left and right could either be the front of the car ot the back of the car
    # careful to use left and right based on image, no of actual car's left and right
    for i in (-1, 1):
        left_constraints.append([left_mult * val_dx, i*val_dy, -switch_mult * val_dz])
    for i in (-1, 1):
        right_constraints.append([right_mult * val_dx, i*val_dy, switch_mult * val_dz])

    # top and bottom are easy, just the top and bottom of car
    for i in (-1, 1):
        for j in (-1, 1):
            top_constraints.append([i*val_dx, -val_dy, j*val_dz])
    for i in (-1, 1):
        for j in (-1, 1):
            bottom_constraints.append([i*val_dx, val_dy, j*val_dz])

    # now, 64 combinations
    for left in left_constraints:
        for top in top_constraints:
            for right in right_constraints:
                for bottom in bottom_constraints:
                    constraints.append([left, top, right, bottom])

    # filter out the ones with repeats
    constraints = filter(lambda x: len(x) == len(set(tuple(i) for i in x)), constraints)

    # create pre M (the term with I and the R*X)
    val_pre_m = np.zeros([4, 4])
    # 1's down diagonal
    for i in range(0, 4):
        val_pre_m[i][i] = 1

    best_loc = None
    best_error = [1e09]
    best_x = None

    # loop through each possible constraint, hold on to the best guess
    # constraint will be 64 sets of 4 corners
    count = 0
    for constraint in constraints:
        # each corner
        val_xa = constraint[0]
        val_xb = constraint[1]
        val_xc = constraint[2]
        val_xd = constraint[3]

        x_array = [val_xa, val_xb, val_xc, val_xd]

        # M: all 1's down diagonal, and upper 3x1 is Rotation_matrix * [x, y, z]
        val_ma = np.copy(val_pre_m)
        val_mb = np.copy(val_pre_m)
        val_mc = np.copy(val_pre_m)
        val_md = np.copy(val_pre_m)

        m_array = [val_ma, val_mb, val_mc, val_md]

        # create A, b
        val_a = np.zeros([4, 3], dtype=np.float)
        val_b = np.zeros([4, 1])

        indicies = [0, 1, 0, 1]
        for row, index in enumerate(indicies):
            x_vals = x_array[row]
            m_vals = m_array[row]

            # create M for corner Xx
            val_rx = np.dot(val_r, x_vals)
            m_vals[:3, 3] = val_rx.reshape(3)

            m_vals = np.dot(proj_matrix, m_vals)

            val_a[row, :] = m_vals[index, :3] - box_corners[row] * m_vals[2, :3]
            val_b[row] = box_corners[row] * m_vals[2, 3] - m_vals[index, 3]

        # solve here with least squares, since over fit will get some error
        loc, error, [], [] = np.linalg.lstsq(val_a, val_b, rcond=None)

        # found a better estimation
        if error < best_error:
            count += 1 # for debugging
            best_loc = loc
            best_error = error
            best_x = x_array

    # return best_loc, [left_constraints, right_constraints] # for debugging
    best_loc = [best_loc[0][0], best_loc[1][0], best_loc[2][0]]
    return best_loc, best_x
