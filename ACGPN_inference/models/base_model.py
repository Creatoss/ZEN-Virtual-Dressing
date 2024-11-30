import os
import torch
import sys

# Base model class that inherits from torch.nn.Module
class BaseModel(torch.nn.Module):
    # Returns the name of the model
    def name(self):
        return 'BaseModel'

    # Initializes the model with given options
    def initialize(self, opt):
        self.opt = opt  # Storing the options passed to the model
        self.gpu_ids = opt.gpu_ids  # GPU IDs for model training (if any)
        self.isTrain = opt.isTrain  # Whether the model is in training mode
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor  # Set tensor type based on GPU availability
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # Directory to save model checkpoints

    # Sets the input to the model
    def set_input(self, input):
        self.input = input

    # Forward pass for the model (should be implemented in subclasses)
    def forward(self):
        pass

    # Method used during testing; no backpropagation
    def test(self):
        pass

    # Placeholder method to get image paths (should be implemented in subclasses)
    def get_image_paths(self):
        pass

    # Placeholder method to optimize model parameters (should be implemented in subclasses)
    def optimize_parameters(self):
        pass

    # Returns the current visuals (usually the input data)
    def get_current_visuals(self):
        return self.input

    # Returns the current errors (an empty dictionary in this base class)
    def get_current_errors(self):
        return {}

    # Placeholder method for saving the model (should be implemented in subclasses)
    def save(self, label):
        pass

    # Helper method to save the model's network
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)  # Constructing file name
        save_path = os.path.join(self.save_dir, save_filename)  # Full save path
        torch.save(network.cpu().state_dict(), save_path)  # Saving model state dict
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda()  # Moving the model back to the GPU if available

    # Helper method to load a saved model's network
    def load_network(self, network, network_label, epoch_label, save_dir=''):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)  # Constructing file name
        print (save_filename)  # Printing the filename for debugging purposes
        if not save_dir:
            save_dir = self.save_dir  # Using default save directory if not provided
        save_path = os.path.join(save_dir, save_filename)  # Full path to saved model
        if not os.path.isfile(save_path):  # If the model file doesn't exist, raise an error
            print('%s not exists yet!' % save_path)
            if network_label == 'G':  # If the generator model is missing, raise an error
                raise('Generator must exist!')
        else:
            network.load_state_dict(torch.load(save_path))  # Loading the saved state dict into the model

    # Placeholder method for updating learning rate (should be implemented in subclasses
            # except:
            #     pretrained_dict = torch.load(save_path)
            #     model_dict = network.state_dict()
            #     try:
            #         pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            #         network.load_state_dict(pretrained_dict)
            #         if self.opt.verbose:
            #             print('Pretrained network %s has excessive layers; Only loading layers that are used' % network_label)
            #     except:
            #         print('Pretrained network %s has fewer layers; The following are not initialized:' % network_label)
            #         for k, v in pretrained_dict.items():
            #             if v.size() == model_dict[k].size():
            #                 model_dict[k] = v
            #
            #         if sys.version_info >= (3,0):
            #             not_initialized = set()
            #         else:
            #             from sets import Set
            #             not_initialized = Set()
            #
            #         for k, v in model_dict.items():
            #             if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
            #                 not_initialized.add(k.split('.')[0])
            #
            #         print(sorted(not_initialized))
            #         network.load_state_dict(model_dict)

    def update_learning_rate():
        pass
