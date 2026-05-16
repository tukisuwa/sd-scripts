# Anima ADDifT training script

from library import train_util
from library.device_utils import init_ipex

init_ipex()

import anima_train_network


def main() -> None:
    parser = anima_train_network.setup_parser()
    args = parser.parse_args()
    if args.ileco:
        raise ValueError("anima_train_addift.py cannot be used with --ileco")
    args.addift = True

    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    if args.dataset_config is None:
        raise ValueError("--dataset_config is required for anima_train_addift.py")
    if args.ileco:
        raise ValueError("anima_train_addift.py cannot be used with --ileco")
    args.addift = True

    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"  # backward compatibility

    trainer = anima_train_network.AnimaNetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
