# Import necessary libraries
import torch.utils.data  # This imports the PyTorch DataLoader utilities
from data.base_data_loader import BaseDataLoader  # Importing the base data loader class for inheritance

# Define the function to create a dataset based on the options passed
def CreateDataset(opt):
    dataset = None  # Initialize dataset variable
    from data.aligned_dataset import AlignedDataset  # Dynamically import AlignedDataset class
    dataset = AlignedDataset()  # Create an instance of AlignedDataset

    # Print confirmation that the dataset was created
    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)  # Initialize the dataset with provided options
    return dataset  # Return the initialized dataset

# Define a custom dataset data loader class that inherits from BaseDataLoader
class CustomDatasetDataLoader(BaseDataLoader):
    # Define the name method to return the name of this data loader
    def name(self):
        return 'CustomDatasetDataLoader'  # Return the name of the class

    # Initialize method to set up dataset and dataloader
    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)  # Call the parent class's initialize method to handle any common setup
        self.dataset = CreateDataset(opt)  # Create and initialize the dataset using the provided options
        # Initialize the DataLoader with specified options
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,  # The dataset to load
            batch_size=opt.batchSize,  # Set batch size from options
            shuffle=not opt.serial_batches,  # Shuffle the data based on the option
            num_workers=int(opt.nThreads))  # Set the number of workers for data loading

    # Method to return the dataloader
    def load_data(self):
        return self.dataloader  # Return the DataLoader object

    # Method to return the length of the dataset or a maximum size, whichever is smaller
    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)  # Return the appropriate dataset size
