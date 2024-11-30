###############################################################################
import torch.utils.data as data
from PIL import Image
import os

# Supported image file extensions
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]

# Function to check if a file is an image based on its extension
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# Function to generate a list of image paths in a directory
def make_dataset(dir):
    images = []
    # Ensure the directory exists
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    # Extract part of the directory name
    f = dir.split('/')[-1].split('_')[-1]
    print(dir, f)

    # Get the list of files in the directory
    dirs = os.listdir(dir)
    for img in dirs:
        # Build the full path of the image file
        path = os.path.join(dir, img)
        # Append the path to the image list
        images.append(path)
    return images

# Function to generate a list of test image paths
def make_dataset_test(dir):
    images = []
    # Ensure the directory exists
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    # Extract part of the directory name
    f = dir.split('/')[-1].split('_')[-1]

    # Loop through the files in the directory
    for i in range(len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])):
        # Depending on the directory name, use different file extensions
        if f == 'label' or f == 'labelref':
            img = str(i) + '.png'
        else:
            img = str(i) + '.jpg'
        # Build the full path of the image file
        path = os.path.join(dir, img)
        # Append the path to the image list
        images.append(path)
    return images

# Default image loader function
def default_loader(path):
    return Image.open(path).convert('RGB')

# Custom dataset class to load images
class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        # Generate image paths using make_dataset
        imgs = make_dataset(root)
        # Raise an error if no images are found
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        # Store parameters
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    # Method to load an image by its index
    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)  # Load the image
        # Apply any transformations to the image
        if self.transform is not None:
            img = self.transform(img)
        # If return_paths is set to True, return image and its path
        if self.return_paths:
            return img, path
        else:
            return img

    # Method to return the number of images in the dataset
    def __len__(self):
        return len(self.imgs)
