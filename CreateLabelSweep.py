import imfusion
import numpy as np
import os


class CreateLabelSweep(imfusion.Algorithm):
    def __init__(self, tracked_stream, untracked_stream):
        super().__init__()
        self.tracked_stream = tracked_stream
        self.untracked_stream = untracked_stream

        self.add_param("flip_image", 'none', attributes = "values: 'none,Y'")

    @classmethod
    def convert_input(cls, data):

        if len(data) != 2:
            raise imfusion.IncompatibleError('Requires two inputs')

        if not isinstance(data[0], imfusion.SharedImageSet) or data[0].modality != imfusion.Data.Modality.ULTRASOUND:
            raise imfusion.IncompatibleError('First input must be a 3D label map')

        if not isinstance(data[1], imfusion.SharedImageSet):
            raise imfusion.IncompatibleError('Second input must be a 2D label map')

        return data

    def compute(self):
        for i, img in enumerate(self.tracked_stream):
            label_array = np.array(self.untracked_stream[i]).copy()
            if self.flip_image == "Y":
                label_array = np.flipud(label_array)

            label_array = label_array.astype(np.uint8)

            arr = np.array(img, copy=False)
            arr[:, :, :] = label_array
            img.setDirtyMem()

        return

