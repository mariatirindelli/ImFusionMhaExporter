import SimpleITK as sitk
import numpy as np
import argparse


class WireEntry:
    def __init__(self, frame_id, n_wire_id, wire_id, row, col ):
        self.frame_id = frame_id
        self.n_wire_id = n_wire_id
        self.wire_id = wire_id
        self.row = row
        self.col = col

    @staticmethod
    def get_from_frame(frame, frame_id, n_wire_id, wire_id, wire_color):
        wire_cluster = np.argwhere(frame == wire_color)
        row_col = np.mean(wire_cluster, axis=0)
        return WireEntry(frame_id=frame_id,
                         n_wire_id=n_wire_id,
                         wire_id=wire_id,
                         row=int(row_col[0]),
                         col=int(row_col[1]))

    def write_on_file(self, fid, delimiter=","):
        fid.write(str(self.frame_id) + delimiter +
                  str(self.n_wire_id) + delimiter +
                  str(self.wire_id) + delimiter +
                  str(self.row) + delimiter +
                  str(self.col))


def process_frame(frame, frame_id):
    n_total_wires = np.max(frame.flatten())

    # If not all wires are in the frame, append -1
    frame_valid = True
    for i in range(1, n_total_wires + 1):
        if np.where( frame.flatten()== i).size == 0:
            frame_valid = False

    wire_entries = []
    if not frame_valid:
        n_nWires = 3 # todo: change this not hard-coded
        for n_wire in range(n_nWires):
            for wire_in_nwire in [1, 2, 3]:
                wire_entries.append(WireEntry(frame_id=frame_id,
                                              n_wire_id=n_wire + 1,
                                              wire_id=wire_in_nwire,
                                              row=-1,
                                              col=-1))
        return wire_entries

    assert n_total_wires % 3 == 0, "Missing wired in the labeling"
    n_nWires = n_total_wires // 3
    for n_wire in range(n_nWires):
        for wire_in_nwire in [1, 2, 3]:
            wire_entries.append(WireEntry.get_from_frame(frame=frame,
                                                         frame_id=frame_id,
                                                         n_wire_id=n_wire + 1,
                                                         wire_id=wire_in_nwire,
                                                         wire_color=n_wire * 3 + wire_in_nwire))

    return wire_entries


def save_wire_entries_as_txt(wire_entries, txt_filepath):
    with open(txt_filepath, 'w') as fid:
        fid.write("#frameId,nWireId,wireId,row,col")
        for wire_entry in wire_entries:
            fid.write("\n")
            wire_entry.write_on_file(fid, delimiter=" ")


def main(input_image_path, output_txt_path):
    sitk_image = sitk.ReadImage(input_image_path)
    image_sequence = sitk.GetArrayFromImage(sitk_image)

    txt_entries = []
    for frame_id in range(image_sequence.shape[0]):
        txt_entries.extend(process_frame(frame=image_sequence[frame_id, ...],
                                         frame_id=frame_id))

    save_wire_entries_as_txt(txt_entries, output_txt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--image_path', type=str,
                        default="/home/maria/Desktop/WirePhantomCalibrationProject/ProcessingPipeline/SegmentedSweeps/wirePhantomSweepsLabels2.mha")
    parser.add_argument('--output_txt_path', type=str,
                        default="/home/maria/Desktop/WirePhantomCalibrationProject/ProcessingPipeline/SegmentedSweeps/wirePhantomFiducials2.txt")

    args = parser.parse_args()
    main(input_image_path=args.image_path,
         output_txt_path=args.output_txt_path)
