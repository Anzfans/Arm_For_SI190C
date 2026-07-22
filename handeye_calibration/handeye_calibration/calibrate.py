import argparse

from handeye_calibration.handeye_solver import main as solve_main
from handeye_calibration.verifier import main as verify_main


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Run hand-eye solve and verification from already collected data.'
    )
    parser.add_argument('--data', default='results/calibration_data.npz')
    parser.add_argument('--transform', default='results/handeye_transform.npy')
    parser.add_argument('--method', default='tsai')
    return parser.parse_args(args)


def main(args=None):
    parsed = parse_args(args)
    print('[Step 1] Solving hand-eye calibration.')
    solve_main([
        '--data', parsed.data,
        '--output', parsed.transform,
        '--method', parsed.method,
    ])
    print('[Step 2] Verifying fixed-target consistency.')
    verify_main([
        '--data', parsed.data,
        '--transform', parsed.transform,
    ])


if __name__ == '__main__':
    main()
