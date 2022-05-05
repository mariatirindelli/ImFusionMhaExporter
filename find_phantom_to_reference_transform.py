import os
os.environ['PATH'] = 'C:\\Program Files\\ImFusion\\ImFusion Suite\\Suite;C:\\Program Files\\ImFusion' \
                     '\\ImFusion Suite\\Suite\\plugins;' + os.environ['PATH']
import numpy as np
import time
# PYTHONUNBUFFERED = 1;PYTHONPATH=C:\Program Files\ImFusion\ImFusion Suite\Suite\;

import imfusion

def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t


def get_tracking_positions(tracking_stream):

    points = []
    for i in range(tracking_stream.size()):
        point = np.squeeze(tracking_stream.matrix(i)[0:3, -1])
        points.append(point)

    return np.stack(points, axis=0)


def average_points_positions(imfusion_file_path):
    imfusion_tracking_streams = imfusion.open(imfusion_file_path)

    point_list = []
    for tracking_stream in imfusion_tracking_streams:
        tracking_point = get_tracking_positions(tracking_stream)
        point_list.append(np.mean(tracking_point, axis=0))

    return point_list


def save_point_cloud(pc, filepath):
    if isinstance(pc, list):
        pc = np.stack(pc, axis=0)

    np.savetxt(filepath, pc)


def find_corresponding_point_transform(p1_path, p2_path):
    pc_1 = np.loadtxt(p1_path)
    pc_2 = np.loadtxt(p2_path)

    R, t = rigid_transform_3D(np.transpose(pc_2), np.transpose(pc_1))

    return R, t


def print_imfusion_matrix(R, t):

    t = t.flatten()
    print_string = "["

    for row in range(3):
        print_string += "["
        for col in range(3):
            print_string += str(R[row, col]) + ", "

        print_string += str(t[row]) + "], "

    print_string += "[0, 0, 0, 1]"

    print(print_string)


def main(calibrated_phantom_point_path, averaged_points_path, stl_points_path):

    # Extracting the phantom landmark positions
    phantom_landmarks = average_points_positions(calibrated_phantom_point_path)
    save_point_cloud(phantom_landmarks, averaged_points_path)

    # Compute the transformation from the phantom .stl model and the acquired landmarks - this is only a sanity
    # check for visualization, it is not needed for the calibration
    R, t = find_corresponding_point_transform(averaged_points_path, stl_points_path)
    print_imfusion_matrix(R, t)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    imfusion.init()

    calibrated_phantom_point_path = "C:/Users/maria/OneDrive/Desktop/wire-phantom-calibration/CalibratedPhantomPoints.imf"
    averaged_points_path = "C:/Users/maria/OneDrive/Desktop/wire-phantom-calibration/averagedPoints.txt"
    stl_points_path = "C:/Users/maria/OneDrive/Desktop/wire-phantom-calibration/phantomStlPoints.txt"
    main(calibrated_phantom_point_path, averaged_points_path, stl_points_path)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
