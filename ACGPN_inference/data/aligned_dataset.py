import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset, make_dataset_test
from PIL import Image
import torch
import json
import numpy as np
import os.path as osp
from PIL import ImageDraw

class AlignedDataset(BaseDataset):
    """
    A dataset class for aligned data used in deep fashion try-on tasks.
    Extends the BaseDataset class to handle multiple input types (label maps, images, masks, etc.).
    """

    def initialize(self, opt):
        """
        Initialize the dataset by setting up paths and configurations for input types.
        Args:
            opt: Options containing dataset paths, modes, and other configurations.
        """
        self.opt = opt
        self.root = opt.dataroot
        self.diction = {}  # Dictionary to index similar images by name

        # Setup for input A (label maps)
        if opt.isTrain or opt.use_encoded_image:
            dir_A = '_A' if self.opt.label_nc == 0 else '_label'
            self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
            self.A_paths = sorted(make_dataset(self.dir_A))
            self.AR_paths = make_dataset(self.dir_A)

        self.fine_height = 256  # Fixed height for image resizing
        self.fine_width = 192   # Fixed width for image resizing
        self.radius = 5         # Radius for pose keypoint visualization

        # Input A setup for testing (label maps)
        if not (opt.isTrain or opt.use_encoded_image):
            dir_A = '_A' if self.opt.label_nc == 0 else '_label'
            self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
            self.A_paths = sorted(make_dataset_test(self.dir_A))
            dir_AR = '_AR' if self.opt.label_nc == 0 else '_labelref'
            self.dir_AR = os.path.join(opt.dataroot, opt.phase + dir_AR)
            self.AR_paths = sorted(make_dataset_test(self.dir_AR))

        # Setup for input B (real images)
        dir_B = '_B' if self.opt.label_nc == 0 else '_img'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.BR_paths = sorted(make_dataset(self.dir_B))
        self.dataset_size = len(self.A_paths)

        # Build an index of similar images
        self.build_index(self.B_paths)

        # Setup for input E (edge maps)
        if opt.isTrain or opt.use_encoded_image:
            dir_E = '_edge'
            self.dir_E = os.path.join(opt.dataroot, opt.phase + dir_E)
            self.E_paths = sorted(make_dataset(self.dir_E))
            self.ER_paths = make_dataset(self.dir_E)

        # Setup for input M (masks)
        if opt.isTrain or opt.use_encoded_image:
            dir_M = '_mask'
            self.dir_M = os.path.join(opt.dataroot, opt.phase + dir_M)
            self.M_paths = sorted(make_dataset(self.dir_M))
            self.MR_paths = make_dataset(self.dir_M)

        # Setup for input MC (color masks)
        if opt.isTrain or opt.use_encoded_image:
            dir_MC = '_colormask'
            self.dir_MC = os.path.join(opt.dataroot, opt.phase + dir_MC)
            self.MC_paths = sorted(make_dataset(self.dir_MC))
            self.MCR_paths = make_dataset(self.dir_MC)

        # Setup for input C (colors)
        if opt.isTrain or opt.use_encoded_image:
            dir_C = '_color'
            self.dir_C = os.path.join(opt.dataroot, opt.phase + dir_C)
            self.C_paths = sorted(make_dataset(self.dir_C))
            self.CR_paths = make_dataset(self.dir_C)

    def random_sample(self, item):
        """
        Randomly samples a new image path similar to the given item.
        Args:
            item: Path of the current image.
        Returns:
            Path of a randomly selected similar image.
        """
        name = item.split('/')[-1].split('-')[0]
        lst = self.diction[name]
        new_lst = [dir for dir in lst if dir != item]
        return new_lst[np.random.randint(len(new_lst))]

    def build_index(self, dirs):
        """
        Builds an index mapping image names to their corresponding file paths for fast retrieval.
        Args:
            dirs: List of image file paths.
        """
        for k, dir in enumerate(dirs):
            name = dir.split('/')[-1].split('-')[0]
            for d in dirs[max(k-20, 0):k+20]:
                if name in d:
                    self.diction.setdefault(name, []).append(d)

    def __getitem__(self, index):
        """
        Fetches a single data point for the dataset.
        Args:
            index: Index of the data point.
        Returns:
            A dictionary containing label maps, real images, masks, edges, poses, etc.
        """
        # Load label maps (A and AR)
        A_path = self.A_paths[index]
        AR_path = self.AR_paths[index]
        A = Image.open(A_path).convert('L')
        AR = Image.open(AR_path).convert('L')

        # Transform label maps
        params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A.convert('RGB'))
            AR_tensor = transform_A(AR.convert('RGB'))
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0
            AR_tensor = transform_A(AR) * 255.0

        # Load real images (B and BR)
        B_path = self.B_paths[index]
        BR_path = self.BR_paths[index]
        B = Image.open(B_path).convert('RGB')
        BR = Image.open(BR_path).convert('RGB')
        transform_B = get_transform(self.opt, params)
        B_tensor = transform_B(B)
        BR_tensor = transform_B(BR)

        # Masks (M) and color masks (MC)
        M = Image.open(B_path).convert('L')
        MR = Image.open(BR_path).convert('L')
        M_tensor = transform_A(MR)
        MC_tensor = transform_A(Image.open(B_path).convert('L'))

        # Load color images (C)
        C = Image.open(self.C_paths[np.random.randint(len(self.C_paths))]).convert('RGB')
        C_tensor = transform_B(C)

        # Edge maps (E)
        E = Image.open(self.E_paths[np.random.randint(len(self.E_paths))]).convert('L')
        E_tensor = transform_A(E)

        # Pose maps
        pose_name = B_path.replace('.jpg', '_keypoints.json').replace('test_img', 'test_pose')
        with open(osp.join(pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = np.array(pose_label['people'][0]['pose_keypoints']).reshape((-1, 3))
        pose_map = torch.zeros(pose_data.shape[0], self.fine_height, self.fine_width)
        im_pose = Image.new('L', (self.fine_width, self.fine_height))
        pose_draw = ImageDraw.Draw(im_pose)
        for i, (x, y, _) in enumerate(pose_data):
            if x > 1 and y > 1:
                pose_draw.rectangle((x-self.radius, y-self.radius, x+self.radius, y+self.radius), 'white', 'white')
            pose_map[i] = transform_B(im_pose.convert('RGB'))[0]

        # Compile input dictionary
        input_dict = {
            'label': A_tensor, 'label_ref': AR_tensor, 'image': B_tensor,
            'image_ref': BR_tensor, 'path': A_path, 'path_ref': AR_path,
            'edge': E_tensor, 'color': C_tensor, 'mask': M_tensor,
            'colormask': MC_tensor, 'pose': pose_map, 'name': B_path.split('/')[-1]
        }
        return input_dict

    def __len__(self):
        """
        Returns the total number of data points in the dataset.
        """
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        """
        Returns the name of the dataset.
        """
        return 'AlignedDataset'
