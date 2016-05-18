import numpy as np
from cv2 import findHomography, warpPerspective


def build_W(points):
    """Build the mean-centered point matrix W. points should be an array with shape
    (num_frames, num_points, 2)."""
    return None


def compute_RS(W):
    """Compute the matrices R and S from W using the SVD."""
    return None


def solve_Q(ih, jh):
    """Solve for Q that satisfies the unit-vector and orthogonality constraints
    on ih and jh."""
    # Build A and b

    # Solve for c via least squares.

    # Form C from c

    # Solve for Q and return it.
    return None


def sfm(points):
    """Run the SfM factorization on a set of points. points will be an array
    with shape (num_frames, num_points, 2)"""
    # Construct the required W/Rh/Sh matrices.

    # Get ih/jh from Rh and use them to find Q.

    # Use Q, Rh, and Sh to get R and S.

    # Extract the rotation matrices from R and put them into a list, one R per
    # image.

    # Return the list of R matrices and an Nx3 matrix P containing the
    # reconstructed 3D points.
    return None


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
