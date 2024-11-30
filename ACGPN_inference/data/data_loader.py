# Define the function to create a data loader
def CreateDataLoader(opt):
    # Dynamically import the CustomDatasetDataLoader class from the specified module
    from data.custom_dataset_data_loader import CustomDatasetDataLoader
    # Instantiate the CustomDatasetDataLoader object
    data_loader = CustomDatasetDataLoader()

    # Print the name of the data loader to confirm it is being created
    print(data_loader.name())

    # Initialize the data loader with the provided options (opt)
    data_loader.initialize(opt)

    # Return the created and initialized data loader
    return data_loader
