import torch  # Importing PyTorch library for tensor computations and model handling


# Function to create a model based on the given options (opt)
def create_model(opt):
    # Check if the model type is 'pix2pixHD' (a specific image-to-image translation model)
    if opt.model == 'pix2pixHD':
        from .pix2pixHD_model import Pix2PixHDModel, \
            InferenceModel  # Importing specific models for training or inference
        if opt.isTrain:  # If the option specifies training mode
            model = Pix2PixHDModel()  # Initialize the model for training
        else:
            model = InferenceModel()  # Initialize the model for inference (testing)

    model.initialize(opt)  # Initialize the model with the provided options (e.g., hyperparameters, configurations)

    # Print the model name if verbose output is enabled
    if opt.verbose:
        print("model [%s] was created" % (model.name()))  # Display the name of the created model for debugging

    # If the model is in training mode and GPU usage is specified, wrap the model in DataParallel for multi-GPU training
    if opt.isTrain and len(opt.gpu_ids):
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)  # Enable parallel processing across multiple GPUs

    return model  # Return the initialized model
