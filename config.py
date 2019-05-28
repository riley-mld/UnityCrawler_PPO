import torch


# Number of batches
NUM_BATCHES = 32
# Discount Factor
GAMMA = 0.99
# Actor learning rate
LR = 1e-4
# Weight decay
WEIGHT_DECAY = 0
# Number of epochs per iteration
EPOCHS = 10
# Maximum number of timesteps per trajector
HORIZON = 2048
# Epsilon for the ratio clip
EPSILON = 0.2
# Entropy coefficient
BETA = 0.0
# Value loss coefficient
C1 = 1.0
# Generalized Advantage Estimate tau coefficient
GAE_TAU = 0.95
# Number of nodes in Actor network
FC1_UNITS = 512
FC2_UNITS = 256
# Clip grad
GRADIENT_CLIP = 0.5


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Configuration():
    """A class to save the hyperparameters and configs."""
    
    def __init__(self):
        """Initialize the class."""
        self.gamma = GAMMA
        self.epochs = EPOCHS
        self.horizon = HORIZON
        self.epsilon = EPSILON
        self.beta = BETA
        self.c1 = C1
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