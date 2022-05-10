import imfusion
import SimpleITK as sitk
import numpy as np


def get_transform_meta(transform, frame_number, transform_name):

    flattened_array = [str(item) for item in transform.flatten().tolist()]
    transform_string = " ".join(flattened_array)

    meta_key = "Seq_Frame" + str(frame_number).zfill(4) + "_" + transform_name + "Transform"
    return meta_key, transform_string


def get_transform_status_meta(transform_name, frame_number):
    meta_key = "Seq_Frame" + str(frame_number).zfill(4) + "_" + transform_name + "TransformStatus"
    return meta_key, "OK"


def get_timestamp_meta(frame_number, timestamp):
    meta_key = "Seq_Frame" + str(frame_number).zfill(4) + "_Timestamps"
    return meta_key, str(timestamp)


def get_image_status_meta(frame_number):
    meta_key = "Seq_Frame" + str(frame_number).zfill(4) + "_ImageStatus"
    return meta_key, "OK"


class TrackingData:
    def __init__(self, frame_idx, timestamp, matrix):
        self.frame_idx = frame_idx
        self.timestamp = timestamp
        self.matrix = matrix


def _us_sweep_tracking_data(imfusion_us_sweep):
    tracking_data = [TrackingData(frame_idx=i,
                                  timestamp=imfusion_us_sweep.timestamp(i),
                                  matrix=imfusion_us_sweep.matrix(i))
                     for i in range(len(imfusion_us_sweep))]
    return tracking_data


def extract_tracking_data(temporal_shared_data):
    # if the image type is US -->
    if isinstance(temporal_shared_data, imfusion.UltrasoundSweep):
        return _us_sweep_tracking_data(temporal_shared_data)

    raise NotImplementedError


class SaveTemporalMha(imfusion.Algorithm):
    def __init__(self, image):
        super().__init__()
        self.add_param('output_filepath', "")
        self.add_param('transform_name', "")
        self.image = image

    @classmethod
    def convert_input(cls, data):

        if len(data) != 1:
            raise imfusion.IncompatibleError('Requires only one input')
        image, = data

        print("Image dimensions: ", image.img().dimensions())

        if image.img().dimensions()[-1] != 1:
            raise imfusion.IncompatibleError("Requires a set of 2D tracked images")

        return {'image': image}

    def compute(self):

        image_data = np.squeeze(self.image)

        # Generating the sitk image from the ImFusion data
        sitk_image = sitk.GetImageFromArray(image_data)

        if self.transform_name == "" and isinstance(self.image, imfusion.UltrasoundSweep):
            instrument_name = self.image.tracking(0).instrumentName
            transform_name = instrument_name + "ToReference"
        else:
            transform_name = self.transform_name if self.transform_name != "" else "ImageToReference"

        # This Meta Key is needed for compatibility with openIgtLink reader from PLUS
        if isinstance(self.image, imfusion.UltrasoundSweep):
            sitk_image.SetMetaData("UltrasoundImageOrientation", "MN")

        # Setting spacing, other meta info (todo)
        sitk_image.SetSpacing(self.image[0].spacing)

        # Extracting the tracking information and adding them to the sitk image as meta keys
        tracking_data = extract_tracking_data(self.image)

        for tracking_sample in tracking_data:
            transform_key, transform_value = get_transform_meta(transform=tracking_sample.matrix,
                                                                frame_number=tracking_sample.frame_idx,
                                                                transform_name=transform_name)

            transform_status_key, transform_status_value = \
                get_transform_status_meta(transform_name=transform_name,
                                          frame_number=tracking_sample.frame_idx)

            timestamp_key, timestamp_value = get_timestamp_meta(frame_number=tracking_sample.frame_idx,
                                                                timestamp=tracking_sample.timestamp)

            image_status_key, image_status_value = get_image_status_meta(frame_number=tracking_sample.frame_idx)

            # Adding all the meta info
            sitk_image.SetMetaData(transform_key, transform_value)
            sitk_image.SetMetaData(transform_status_key, transform_status_value)
            sitk_image.SetMetaData(timestamp_key, timestamp_value)
            sitk_image.SetMetaData(image_status_key, image_status_value)

        sitk.WriteImage(sitk_image, self.output_filepath)
        print("Meta image Successfully saved in: ", self.output_filepath)

