import argparse
from SaveTemporalMha import *


def main(input_imf_sweep_path, sweep_id, output_mha_path, transform_name):

    imfusion.init()
    imfusion.registerAlgorithm('Export Meta Sequence', SaveTemporalMha)

    imfusion_sweep = imfusion.open(input_imf_sweep_path)[sweep_id]
    properties = imfusion.Properties({'output_filepath': output_mha_path, 'transform_name': transform_name})

    imfusion.executeAlgorithm('Export Meta Sequence', [imfusion_sweep], properties)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input_path', type=str,
                        default="/home/maria/Desktop/wire/cephasonics_15cm_LabelSweep.imf")
    parser.add_argument('--sweep_id', type=int,
                        default=0)

    parser.add_argument('--output_path', type=str,
                        default="/home/maria/Desktop/wire/cephasonics_15cm_LabelSweep.mha")
    parser.add_argument('--transform_name', type=str,
                        default="ProbeToTracker")

    args = parser.parse_args()
    main(input_imf_sweep_path=args.input_path,
         sweep_id=args.sweep_id,
         output_mha_path=args.output_path,
         transform_name=args.transform_name)
