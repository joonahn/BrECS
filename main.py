#!/usr/bin/env python3
import os
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter
from glob import glob
from natsort import os_sorted
from data import DataScheduler
from models import MODEL
from train import train_model
from utils.arguments import parse_args
from utils.util import set_seed_everywhere, count_parameters

def main():
	args = parse_args()

	# Load config
	config_path = args.config

	if args.resume_latest_ckpt:
		log_dir = args.resume_latest_ckpt
		assert os.path.isdir(log_dir), 'log_dir {} does not exist'.format(log_dir)
		latest_path = os_sorted(glob(os.path.join(log_dir, 'ckpts/*')))[-1]
		print('Loading latest checkpoint: {}'.format(latest_path))
		base_dir = os.path.dirname(os.path.dirname(latest_path))
		config_path = os.path.join(base_dir, 'config.yaml')
		args.resume_ckpt = latest_path
	elif args.resume_ckpt:
		base_dir = os.path.dirname(os.path.dirname(args.resume_ckpt))
		config_path = os.path.join(base_dir, 'config.yaml')
	config = yaml.load(open(config_path), Loader=yaml.FullLoader)

	# Override options
	for option in args.override.split('|'):
		if not option:
			continue
		address, value = option.split('=')
		keys = address.split('.')
		here = config
		for key in keys[:-1]:
			if key not in here:
				raise ValueError('{} is not defined in config file. '
								 'Failed to override.'.format(address))
			here = here[key]
		if keys[-1] not in here:
			raise ValueError('{} is not defined in config file. '
							 'Failed to override.'.format(address))
		here[keys[-1]] = yaml.load(value, Loader=yaml.FullLoader)

	# Set log directory
	config['log_dir'] = args.log_dir
	if not args.resume_ckpt and os.path.exists(args.log_dir):
		print('WARNING: %s already exists' % args.log_dir)
		input('Press enter to continue')
		print()

	if args.resume_ckpt and not args.log_dir:
		config['log_dir'] = os.path.dirname(
			os.path.dirname(args.resume_ckpt)
		)

	# Save config
	os.makedirs(config['log_dir'], mode=0o755, exist_ok=True)
	if (not args.resume_ckpt or args.config) and (not args.test):
		config_save_path = os.path.join(config['log_dir'], 'config.yaml')
		yaml.dump(config, open(config_save_path, 'w'))
		print('Config saved to {}'.format(config['log_dir']))

	# set random seed
	if config['seed'] is not None:
		set_seed_everywhere(config['seed'])

	# Build components
	writer = SummaryWriter(config['log_dir'])
	data_scheduler = DataScheduler(config)
	model = MODEL[config['model']](config, writer)

	print('# of parameters: {}'.format(count_parameters(model)))

	model.to(config['device'])
	if args.resume_ckpt:
		# load trained model
		checkpoint = torch.load(args.resume_ckpt)
		model.load_state_dict(checkpoint['model_state_dict'])
		model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		if checkpoint.get('lr_scheduler_state_dict') is not None:
			model.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
		resume_step = checkpoint['step'] + 1
		if (args.test is not None) and config.get('cache_only'):
			resume_step -= 1
	else:
		resume_step = 0

	if args.resume_ckpt:
		data_scheduler.epoch_cnt = checkpoint['epoch']

	if args.test:
		if args.single_iou:
			data_scheduler.measure_single_iou(model, args.n_iou_data)
		elif args.multi_iou:
			pass
		else:
			data_scheduler.test(model, writer, resume_step)
	else:
		train_model(config, model, data_scheduler, writer, resume_step)


if __name__ == '__main__':
	main()
