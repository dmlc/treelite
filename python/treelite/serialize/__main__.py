# coding: utf-8
"""
CLI script to convert a model file to a Treelite binary checkpoint
"""
import argparse
import os
import pickle
import treelite


def main():
    """Convert a model file to a Treelite binary checkpoint"""
    parser = argparse.ArgumentParser(description='CLI script to convert a model file to a ' +
                                                 'Treelite binary checkpoint')
    parser.add_argument('--input-model', type=str, required=True,
                        help='Path to the tree model file')
    parser.add_argument('--input-model-type', type=str, required=True,
                        choices=['sklearn_pkl', 'xgboost', 'xgboost_json', 'lightgbm'],
                        help='Type of the tree model file')
    parser.add_argument('--output-checkpoint', type=str, required=True,
                        help='Path to the checkpoint file, using a special binary format for ' +
                             'Treelite')
    args = parser.parse_args()

    if args.input_model_type == 'sklearn_pkl':
        with open(args.input_model, 'rb') as f:
            sklearn_model = pickle.load(f)
        model = treelite.sklearn.import_model(sklearn_model)
    elif args.input_model_type in ['xgboost', 'xgboost_json', 'lightgbm']:
        model = treelite.Model.load(args.input_model, model_format=args.input_model_type)
    else:
        raise ValueError(f'Unrecognized type of model type: {args.input_model_type}')
    dest_checkpoint = os.fspath(os.path.expanduser(args.output_checkpoint))
    model.serialize(dest_checkpoint)


if __name__ == '__main__':
    main()
