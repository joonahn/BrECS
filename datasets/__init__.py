from datasets.lego_modelnet_dataset import (
	LegoModelnetDataset
)

DATASET = {
	# # autoencoder datasets
	# AutoencoderShapenetDataset.name: AutoencoderShapenetDataset,

	# # transition datasets
	# TransitionShapenetDataset.name: TransitionShapenetDataset,
	# TransitionSyntheticRoomDataset.name: TransitionSyntheticRoomDataset,

	# lego datasets
	LegoModelnetDataset.name: LegoModelnetDataset
}
