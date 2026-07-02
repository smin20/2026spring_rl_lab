import argparse
import importlib
import os


TRAINER_MAP = {
    'dqn': 'DQNTrainer',
    'ddpg': 'DDPGTrainer',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--algo',
        choices=list(TRAINER_MAP.keys()),
        required=True,
        help='Choose algorithm to run. Choices: [dqn, ddpg]',
    )
    parser.add_argument('--map', type=str, default='map1', help='Map to run. Choices: [map0, map1, map2, map3]')
    parser.add_argument('--save_name', type=str, default='example', help='Filename of saving policy to pth file')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--logdir', type=str, default='runs')
    parser.add_argument(
        '--step-size',
        type=float,
        default=None,
        help='Movement length in meters. Defaults: dqn=0.2, ddpg=1.0',
    )

    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--heatmap-interval', type=int, default=100)
    parser.add_argument('--resolution', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.step_size is None:
        args.step_size = 1.0 if args.algo == 'ddpg' else 0.2

    os.makedirs(args.logdir, exist_ok=True)

    mod = importlib.import_module(f'algos.{args.algo}')
    trainer_class = getattr(mod, TRAINER_MAP[args.algo])
    trainer = trainer_class(args)

    trainer.initialize()
    trainer.train()
    trainer.save()


if __name__ == '__main__':
    main()
