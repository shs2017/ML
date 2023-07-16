import argparse

def arguments():
    parser = argparse.ArgumentParser(description='Trains a transformer')
    parser.add_argument('-c', '--checkpoint_path')
    return parser.parse_args()
