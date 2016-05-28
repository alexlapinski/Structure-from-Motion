import numpy as np
from cv2 import findHomography, warpPerspective


def build_W(points, debug=False):
    """Build the mean-centered point matrix W. points should be an array with shape
    (num_frames, num_points, 2)."""

    num_frames = points.shape[0]
    num_points = points.shape[1]

    num_rows = 2 * num_frames
    num_cols = num_points
    normalized_W = np.zeros((num_rows, num_cols))

    # Points are in the format X Y
    for f in xrange(len(points)):
        frame = points[f]
        #x_total = 0
        #y_total = 0
        #for point in frame:
        #    x_total += point[0]
        #    y_total += point[1]
        #x_average = x_total / len(frame)
        #y_average = y_total / len(frame)
        mean = frame.mean(0)

        print mean

        for p in xrange(len(frame)):
            point = frame[p]
            normalized_W[f][p] = point[0] - mean[0]
            normalized_W[f + num_frames][p] = point[1] - mean[1]

    return normalized_W


def compute_RS(W):
    """Compute the matrices R and S from W using the SVD."""

    U, d, V = np.linalg.svd(W)

    # Take the 1d array 'd' and put it into the correct diagonal shape
    D = np.diag(d)

    # Cut down the factorized matricies to correspond with rank 3
    U_prime = U[:, :3]
    D_prime = D[:3, :3]
    V_prime = V[:3, :]

    # Apply sqrt of D to U and V
    half_D_prime = np.sqrt(D_prime)
    R = np.dot(U_prime, half_D_prime)
    S = np.dot(half_D_prime, V_prime)

    return R, S


def solve_Q(i_hat, j_hat):
    """Solve for Q that satisfies the unit-vector and orthogonality constraints
    on ih and jh."""

    if i_hat.shape[0] != j_hat.shape[0] or i_hat.shape[1] != j_hat.shape[1]:
        raise ValueError("i_hat's shape must equal j_hat's shape")

    num_frames = i_hat.shape[0]

    # Build A and b
    A = np.zeros((3 * num_frames, 6))

    offset = 0
    for frame in xrange(num_frames):
        i_hat_row = i_hat[frame]
        j_hat_row = j_hat[frame]

        # There probably is a 'better' way to compute each row segment
        # but this works for now
        elem_1_1 = i_hat_row[0]**2
        elem_1_2 = 2 * i_hat_row[0] * i_hat_row[1]
        elem_1_3 = 2 * i_hat_row[0] * i_hat_row[2]
        elem_1_4 = i_hat_row[1]**2
        elem_1_5 = 2 * i_hat_row[1] * i_hat_row[2]
        elem_1_6 = 2 * i_hat_row[2]**2
        row_1 = [elem_1_1, elem_1_2, elem_1_3, elem_1_4, elem_1_5, elem_1_6]

        elem_2_1 = j_hat_row[0]**2
        elem_2_2 = 2 * j_hat_row[0] * j_hat_row[1]
        elem_2_3 = 2 * j_hat_row[1] * j_hat_row[2]
        elem_2_4 = j_hat_row[1]**2
        elem_2_5 = 2 * j_hat_row[1] * j_hat_row[2]
        elem_2_6 = j_hat_row[2]**2
        row_2 = [elem_2_1, elem_2_2, elem_2_3, elem_2_4, elem_2_5, elem_2_6]

        elem_3_1 = i_hat_row[0] * j_hat_row[0]
        elem_3_2 = i_hat_row[0] * j_hat_row[1] + i_hat_row[1] * j_hat_row[0]
        elem_3_3 = i_hat_row[0] * j_hat_row[2] + i_hat_row[2] * j_hat_row[0]
        elem_3_4 = i_hat_row[1] * j_hat_row[1]
        elem_3_5 = i_hat_row[1] * j_hat_row[2]
        elem_3_6 = i_hat_row[2] * j_hat_row[2]
        row_3 = [elem_3_1, elem_3_2, elem_3_3, elem_3_4, elem_3_5, elem_3_6]

        A[offset:offset + 3] = [row_1, row_2, row_3]
        offset += 3

    b = np.ones((3 * num_frames))
    # Every 3rd row is equal to zero
    b[2::3] = 0

    # Solve for c via least squares.
    result = np.linalg.lstsq(A, b)
    c = result[0]

    # Form C from c
    C = np.zeros((3, 3))
    C[0] = [c[0], c[1], c[2]]
    C[1] = [c[1], c[3], c[4]]
    C[2] = [c[2], c[4], c[5]]

    # Solve for Q and return it.
    Q = np.linalg.cholesky(C)

    return Q


def sfm(points):
    """Run the SfM factorization on a set of points. points will be an array
    with shape (num_frames, num_points, 2)"""

    num_frames = points.shape[0]

    # Construct the required W/Rh/Sh matrices.
    normalized_W_matrix = build_W(points)
    R_hat, S_hat = compute_RS(normalized_W_matrix)

    # Get ih/jh from Rh and use them to find Q.
    i_hat = R_hat[:num_frames, :]
    j_hat = R_hat[num_frames:, :]

    Q = solve_Q(i_hat, j_hat)

    # Use Q, Rh, and Sh to get R and S.
    R = np.dot(R_hat, Q)
    S = np.linalg.solve(Q, S_hat)

    print "R.shape: ", R.shape, "; S.shape: ", S.shape

    # Extract the rotation matrices from R and put them into a list, one R per
    # image.
    rotation_matricies = list()

    offset = 0
    for frame in xrange(num_frames):
        rotation_matrix = np.zeros((2, R.shape[1]))
        temp = R[offset:offset + 2]
        print "Frame: ", frame, "temp.shape", temp.shape
        rotation_matrix[:] = R[offset:offset + 2]
        offset += 2
        rotation_matricies.append(rotation_matrix)

    # Return the list of R matrices and an Nx3 matrix P containing the
    # reconstructed 3D points.
    print "S", S
    print "S.T", S.T
    return rotation_matricies, S.T


def get_texture(images, region_points, texture_size=256):
    """Given a set of images and 4 points for each image representing a
    quadrilateral planar region, extract a square texture of size texture_size
    containing the pixel values within the region averaged over all of the
    images You may use the imported OpenCV findHomography and warpPerspective
    functions."""

    # Build a (4,2) array of X/Y texture coordinates for a
    # texture_size x texture_size square. The coordinates should
    # start at the top left (0,0) and proceed clockwise.

    for image, rect_points in zip(images, region_points):
        # Find a homography that warps the points for the current region to the
        # texture coordinates.

        # Warp the image with the homography to obtain the texture for this
        # image and append it to the list of textures.
        pass

    # Return the mean texture across the images.
    return None
