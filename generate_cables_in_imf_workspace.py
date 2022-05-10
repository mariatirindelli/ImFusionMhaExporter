import numpy as np
import xml.etree.ElementTree as ET
import random
from scipy.spatial.transform import Rotation as R
import imfusion


class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def transform(self, T):
        homogeneous_point = np.array([[self.x], [self.y], [self.z], [1]])
        transformed_point = np.matmul(T, homogeneous_point)
        self.x = transformed_point[0]
        self.y = transformed_point[1]
        self.z = transformed_point[2]

    def as_array(self):
        return np.array([self.x, self.y, self.z])


class Wire:
    def __init__(self, start_point, end_point, name = ""):
        self.start_point = start_point
        self.end_point = end_point
        self.name = name

    def transform(self, T):
        self.start_point.transform(T)
        self.end_point.transform(T)

    @staticmethod
    def get_from_config_block(config_block):
        # <Wire Name="1:D9_d9" EndPointFront="-10.0 0.0 -50.0" EndPointBack="-10.0 40.0 -50.0" />

        name = config_block.attrib["Name"]
        end_point_front_string = config_block.attrib["EndPointFront"]
        end_point_back_string = config_block.attrib["EndPointBack"]

        end_point_front = [float(item) for item in end_point_front_string.split(" ") if item != " " and item != ""]
        end_point_back = [float(item) for item in end_point_back_string.split(" ") if item != " " and item != ""]

        return Wire(start_point=Point(x=end_point_front[0], y=end_point_front[1], z=end_point_front[2]),
                    end_point=Point(x=end_point_back[0], y=end_point_back[1], z=end_point_back[2]),
                    name=name)


# intersection function
def get_plane_wire_intersection(p0, p1, p_co, p_no, epsilon=1e-6):
    """
    p0, p1: Define the line.
    p_co, p_no: define the plane:
        p_co Is a point on the plane (plane coordinate).
        p_no Is a normal vector defining the plane direction;
             (does not need to be normalized).

    Return a Vector or None (when the intersection can't be found).
    """

    u = sub_v3v3(p1, p0)
    dot = dot_v3v3(p_no, u)

    if abs(dot) > epsilon:
        # The factor of the point between p0 -> p1 (0 - 1)
        # if 'fac' is between (0 - 1) the point intersects with the segment.
        # Otherwise:
        #  < 0.0: behind p0.
        #  > 1.0: infront of p1.
        w = sub_v3v3(p0, p_co)
        fac = -dot_v3v3(p_no, w) / dot
        u = mul_v3_fl(u, fac)
        return add_v3v3(p0, u)

    # The segment is parallel to plane.
    return None

# ----------------------
# generic math functions

def add_v3v3(v0, v1):
    return (
        v0[0] + v1[0],
        v0[1] + v1[1],
        v0[2] + v1[2],
    )


def sub_v3v3(v0, v1):
    return (
        v0[0] - v1[0],
        v0[1] - v1[1],
        v0[2] - v1[2],
    )


def dot_v3v3(v0, v1):
    return (
        (v0[0] * v1[0]) +
        (v0[1] * v1[1]) +
        (v0[2] * v1[2])
    )


def len_squared_v3(v0):
    return dot_v3v3(v0, v0)


def mul_v3_fl(v0, f):
    return (
        v0[0] * f,
        v0[1] * f,
        v0[2] * f,
    )


def get_geometry_block(root):
    for element0 in root:
        if element0.tag != "PhantomDefinition":
            continue

        for element1 in element0:
            if element1.tag == "Geometry":
                return element1


def get_wires(config_path):
    tree = ET.parse(config_path)
    root = tree.getroot()

    geometry_block = get_geometry_block(root)

    wires = []
    for item in geometry_block:
        if item.tag != "Pattern":
            continue
        for wire_block in item:
            wires.append(Wire.get_from_config_block(wire_block))

    return wires


def get_roi(wires, tolerance = 0, rotation_range = 0):

    all_wire_points = []
    for item in wires:
        all_wire_points.append(item.start_point)
        all_wire_points.append(item.end_point)

    min_x = np.min([item.x for item in all_wire_points]) - tolerance
    max_x = np.max([item.x for item in all_wire_points]) + tolerance

    min_y = np.min([item.y for item in all_wire_points]) - tolerance
    max_y = np.max([item.y for item in all_wire_points]) + tolerance

    min_z = np.min([item.z for item in all_wire_points]) - tolerance
    max_z = np.max([item.z for item in all_wire_points]) + tolerance

    # rotation : +-20 on each axis
    return [min_x, max_x], [min_y, max_y], [min_z, max_z], [-rotation_range, rotation_range], \
           [-rotation_range, rotation_range], [-rotation_range, rotation_range]


def get_matrix_from_string(matrix_string):
    mat_num = [float(item) for item in matrix_string.split(" ") if item != " " and item != ""]
    mat_reshaped = np.reshape(mat_num, [4, 4])
    return mat_reshaped


def get_phantom_to_reference(config_path):

    tree = ET.parse(config_path)
    root = tree.getroot()
    for element0 in root:
        if element0.tag != "CoordinateDefinitions":
            continue

        for element1 in element0:
            if element1.tag != "Transform":
                continue

            if element1.attrib["From"] == "Phantom" and  element1.attrib["To"] == "Tracker":
                return get_matrix_from_string(element1.attrib["Matrix"])


def generate_image_trajectory(roi, n_images = 200):

    transform_list = []
    for i in range(n_images):
        x_t = np.random.uniform(roi[0][0], roi[0][1])
        y_t = np.random.uniform(roi[1][0], roi[1][1])
        z_t = np.random.uniform(roi[2][0], roi[2][1])

        x_r = np.random.uniform(roi[3][0], roi[3][1])
        y_r = np.random.uniform(roi[4][0], roi[4][1])
        z_r = np.random.uniform(roi[5][0], roi[5][1])

        r = R.from_euler("zyx", [z_r, y_r, x_r], degrees=True)
        t = np.eye(4)
        t[0:3, 0:3] = r.as_matrix()
        t[0:3, -1] = np.array([x_t, y_t, z_t])

        transform_list.append(t)

    return transform_list


def get_intersection_on_image(intersection, T_imageToTracker, spacing_x, spacing_y):
    p_hom = np.array([[intersection[0][0]], [intersection[1][0]], [intersection[2][0]], [1]])
    p_img = np.matmul(np.linalg.inv(T_imageToTracker), p_hom)

    return int(p_img[0]/spacing_x), int(p_img[1]/spacing_y)


def generate_sweep_frame(image_intersections, image_size, draw_size = 3):
    frame = np.zeros(image_size)
    for i, intersection in enumerate(image_intersections):
        col = intersection[0]
        row = intersection[1]

        frame[max(row-draw_size, 0):min(row+draw_size, image_size[0]), max(col-draw_size, 0):min(col+draw_size, image_size[1])] = i + 1

    return frame


def generate_imfusion_sweep(image_frames, image_transforms, spacing):
    shared_image_set = imfusion.SharedImageSet()

    for image in image_frames:
        current_frame = np.expand_dims(image, axis=-1).astype(np.uint8)
        shared_us_image = imfusion.SharedImage(current_frame)
        shared_us_image.spacing = np.array(spacing)
        shared_image_set.add(shared_us_image)

    tracking_stream = imfusion.TrackingStream()
    for transform in image_transforms:

        col0 = transform[:, 0]
        col1 = transform[:, 1]
        col2 = transform[:, 2]
        # transform[:, 0] = col2
        # transform[:, 1] = col0
        # transform[:, 2] = col2

        tracking_stream.add(transform)

    shared_image_set.modality = imfusion.Data.Modality.ULTRASOUND

    # a = imfusion.open("/home/maria/Desktop/wire/cephasonics 15cm.imf")[0]
    # b = np.array(a[0])

    imfusion.executeAlgorithm('IO;Tracking Stream', [tracking_stream],
                          imfusion.Properties(
                              {
                                  'location': "tracking_stream.ts"}))

    output_sweep = imfusion.executeAlgorithm("Convert to Sweep", [shared_image_set],
                                           imfusion.Properties({'Tracking stream file': "tracking_stream.ts"}))

    return output_sweep


def main(config_path, image_spacing_x, image_spacing_y, image_size_x, image_size_y, n_wires = 9):

    # todo: this we can set somehow
    T_imageToProbe = np.eye(4)

    T_phantomToReference = get_phantom_to_reference(config_path)

    wires = get_wires(config_path)
    _ = [item.transform(T_phantomToReference) for item in wires]

    # todo generate trajectory automatically
    roi = get_roi(wires)
    T_imageToTracker = generate_image_trajectory(roi)

    image_frames = []
    image_transforms = []
    for transform in T_imageToTracker:
        image_intersection = []
        for wire in wires:

            intersection = get_plane_wire_intersection(wire.start_point.as_array(),
                                                       wire.end_point.as_array(),
                                                       transform[0:3, -1],
                                                       transform[0:3, 2])

            intersection_on_image = get_intersection_on_image(intersection, transform, image_spacing_x, image_spacing_y)

            if intersection_on_image[0] < 0 or intersection_on_image[1] > image_size_x or \
                    intersection_on_image[1] < 0 or intersection_on_image[1] > image_size_y:
                continue

            image_intersection.append(intersection_on_image)

        if len(image_intersection) < n_wires:
            continue

        frame = generate_sweep_frame(image_intersection, [image_size_y, image_size_x])
        T_probeToTracker = np.matmul(transform, np.linalg.inv(T_imageToProbe))

        image_frames.append(frame)
        image_transforms.append(T_probeToTracker)
        # todo: update transform in a .mha file

        #todo: add intersection to the fiducials txt file

    imfusion_sweep = generate_imfusion_sweep(image_frames=image_frames, image_transforms=image_transforms,
                                             spacing=[image_spacing_x, image_spacing_y, 1])

    imfusion.executeAlgorithm('IO;ImFusionFile', imfusion_sweep,
                              imfusion.Properties({'location': "tmp1.imf"}))

    return

imfusion.init()
main("/home/maria/imfusion/plus-calibration/config_2.xml", 0.5, 0.5, 400, 600)