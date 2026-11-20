"""Small stable CLI dispatcher; adapters own their native result schemas."""
import argparse


def select_algorithm(arguments):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--algorithm', choices=['legacy_g99', 'tum_native'], default='legacy_g99')
    options, remaining = parser.parse_known_args(arguments)
    return options.algorithm, remaining
