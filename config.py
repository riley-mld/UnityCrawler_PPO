import torch


# Number of batches
NUM_BATCHES = 32
# Discount Factor
GAMMA = 0.99
# Actor learning rate
LR = 1e-3
# Weight decay
WEIGHT_DECAY
# Number of epochs per iteration
EPOCH = 10
# Maximum number of timesteps per trajector
HORIZON = 100
# Epsilon for the ratio clip
EPSILON = 0.2
# Entropy coefficient
BETA = 0.01
# Generalized Advantage Estimate tau coefficient
GAE_TAU = 0.95
# Number of nodes in Actor network
FC1_UNITS = 256
FC2_UNITS = 256
# Clip grad
GRADIENT_CLIP = 5


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Configuration():
    """A class to save the hyperparameters and configs."""
    
    def __init__(self):
        """Initialize the class."""
        self.gamma = GAMMA
        self.epoch = EPOCH
        self.horizon = HORIZON
        self.epsilon = EPSILON
        self.beta = BETA
        self.gae_tau = GAE_TAU
        self.num_batches = NUM_BATCHES
        
        # Network params
        self.fc1_units = FC1_UNITS
        self.fc2_units = FC2_UNITS
        self.weight_decay = WEIGHT_DECAY
        self.lr = LR
        self.gradient_clip = GRADIENT_CLIP
        
        # Set training device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")