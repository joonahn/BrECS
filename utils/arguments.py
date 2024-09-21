import os
from argparse import ArgumentParser
from datetime import datetime, timezone, timedelta

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--config', '-c',
        default='configs/cgca_autoencoder-circle.yaml'
    )
    parser.add_argument(
        '--log-dir', '-l',
        default=None
    )
    parser.add_argument('--resume-ckpt')
    parser.add_argument('--resume-latest-ckpt')
    parser.add_argument('--override', default='')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--single_iou', default=False, action='store_true')
    parser.add_argument('--multi_iou', default=False, action='store_true')
    parser.add_argument('--n_iou_data', default=None, type=int)

    args = parser.parse_args()

    if args.log_dir == None:
        KST = timezone(timedelta(hours=9))
        ts = datetime.now(KST).strftime('%m-%d-%H:%M:%S')
        expname = os.path.splitext(os.path.basename(args.config))[0]
        args.log_dir = f"./log/{ts}-{expname}"

    return args