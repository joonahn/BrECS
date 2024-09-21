from models.gca import GCA
from models.cgca_transition import CGCATransitionModel, CGCATransitionConditionModel
from models.cgca_autoencoder import CGCAAutoencoder
from models.lego_model import LegoTransitionModel

MODEL = {
    GCA.name: GCA,
	CGCATransitionModel.name: CGCATransitionModel,
	CGCATransitionConditionModel.name: CGCATransitionConditionModel,
	CGCAAutoencoder.name: CGCAAutoencoder,
	LegoTransitionModel.name: LegoTransitionModel,

}

