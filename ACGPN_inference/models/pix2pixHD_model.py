import numpy as np  # Importing NumPy for numerical operations, particularly matrix operations
import torch  # Importing PyTorch for deep learning and tensor manipulation
import os  # For interacting with the operating system (like file operations)
from torch.autograd import Variable  # Importing Variable to wrap tensor operations for automatic differentiation
from util.image_pool import ImagePool  # Importing ImagePool, a utility for maintaining a pool of images for training (often used in GANs)
import torch.nn as nn  # Importing the neural network module from PyTorch

import cv2  # OpenCV library used for image processing tasks
from .base_model import BaseModel  # Importing BaseModel, possibly a custom class for common model functionality
from . import networks  # Importing networks module, likely containing definitions for the neural network architectures
import torch.nn.functional as F  # Importing functional API of PyTorch for commonly used operations like activation functions, etc.

NC = 20  # Setting a constant for the number of classes (likely used for segmentation tasks)

# Function to generate a discrete label for input tensor based on the label dimensions
def generate_discrete_label(inputs, label_nc, onehot=True, encode=True):
    pred_batch = []
    size = inputs.size()  # Getting the size of the input tensor
    for input in inputs:
        input = input.view(1, label_nc, size[2], size[3])  # Reshaping input tensor to a 4D shape
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)  # Getting the max index per class
        pred_batch.append(pred)  # Storing the discrete label predictions

    pred_batch = np.array(pred_batch)  # Converting the list to a numpy array
    pred_batch = torch.from_numpy(pred_batch)  # Converting it back to a PyTorch tensor
    label_map = []
    for p in pred_batch:
        p = p.view(1, 256, 192)  # Reshaping the predictions
        label_map.append(p)
    label_map = torch.stack(label_map, 0)  # Stacking all predictions to form the final label map

    # One-hot encoding the label map if needed
    if not onehot:
        return label_map.float().cuda()  # Returning the label map directly
    size = label_map.size()
    oneHot_size = (size[0], label_nc, size[2], size[3])
    input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()  # Creating a zero tensor for one-hot encoding
    input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)  # One-hot encoding the label

    return input_label

# Morphological operation function: dilation or erosion to modify mask shapes
def morpho(mask, iter, bigger=True):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # Defining a 3x3 elliptical kernel for morphological operations
    new = []
    for i in range(len(mask)):
        tem = mask[i].cpu().detach().numpy().squeeze().reshape(256, 192, 1) * 255  # Reshaping mask to image format
        tem = tem.astype(np.uint8)  # Converting to unsigned int type for OpenCV processing
        if bigger:
            tem = cv2.dilate(tem, kernel, iterations=iter)  # Dilation operation
        else:
            tem = cv2.erode(tem, kernel, iterations=iter)  # Erosion operation
        tem = tem.astype(np.float64)
        tem = tem.reshape(1, 256, 192)
        new.append(tem.astype(np.float64) / 255.0)  # Normalizing back to [0, 1] range
    new = np.stack(new)  # Stacking the modified masks
    new = torch.FloatTensor(new).cuda()  # Converting back to tensor and transferring to GPU
    return new

# Similar function with a smaller kernel (1x1) for finer morphological operations
def morpho_smaller(mask, iter, bigger=True):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))  # Using a smaller kernel
    new = []
    for i in range(len(mask)):
        tem = mask[i].cpu().detach().numpy().squeeze().reshape(256, 192, 1) * 255
        tem = tem.astype(np.uint8)
        if bigger:
            tem = cv2.dilate(tem, kernel, iterations=iter)  # Dilation
        else:
            tem = cv2.erode(tem, kernel, iterations=iter)  # Erosion
        tem = tem.astype(np.float64)
        tem = tem.reshape(1, 256, 192)
        new.append(tem.astype(np.float64) / 255.0)  # Normalizing back to [0, 1] range
    new = np.stack(new)
    new = torch.FloatTensor(new).cuda()
    return new

# Function to encode the label map into one-hot format
def encode(label_map, size):
    label_nc = 14  # Number of classes
    oneHot_size = (size[0], label_nc, size[2], size[3])  # Defining the size for one-hot encoding
    input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()  # Creating a zero tensor for encoding
    input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)  # One-hot encoding
    return input_label

# The Pix2PixHD model class that inherits from the BaseModel class
class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'  # Returning the model name

    # Function to filter which losses to use based on flags
    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):
        flags = (True, use_gan_feat_loss, use_vgg_loss, True, True)

        def loss_filter(g_gan, g_gan_feat, g_vgg, d_real, d_fake):
            return [l for (l, f) in zip((g_gan, g_gan_feat, g_vgg, d_real, d_fake), flags) if f]

        return loss_filter

    # Function to define the Generator network
    def get_G(self, in_C, out_c, n_blocks, opt, L=1, S=1):
        return networks.define_G(in_C, out_c, opt.ngf, opt.netG, L, S,
                                 opt.n_downsample_global, n_blocks, opt.n_local_enhancers,
                                 opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)

    # Function to define the Discriminator network
    def get_D(self, inc, opt):
        netD = networks.define_D(inc, opt.ndf, opt.n_layers_D, opt.norm, opt.no_lsgan,
                                 opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
        return netD

    # Cross-entropy loss function for segmentation
    def cross_entropy2d(self, input, target, weight=None, size_average=True):
        n, c, h, w = input.size()
        nt, ht, wt = target.size()

        if h != ht or w != wt:
            input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)  # Resize input if needed

        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)  # Reshaping for cross-entropy loss
        target = target.view(-1)
        loss = F.cross_entropy(
            input, target, weight=weight, size_average=size_average, ignore_index=250
        )

        return loss

    # Function to compute the average color from a mask and corresponding arms image
    def ger_average_color(self, mask, arms):
        color = torch.zeros(arms.shape).cuda()
        for i in range(arms.shape[0]):
            count = len(torch.nonzero(mask[i, :, :, :]))  # Counting the non-zero pixels in the mask
            if count < 10:
                color[i, 0, :, :] = 0
                color[i, 1, :, :] = 0
                color[i, 2, :, :] = 0  # If there are too few non-zero pixels, set color to black
            else:
                color[i, 0, :, :] = arms[i, 0, :, :].sum() / count  # Averaging color for each channel
                color[i, 1, :, :] = arms[i, 1, :, :].sum() / count
                color[i, 2, :, :] = arms[i, 2, :, :].sum() / count
        return color

    # Initialization function for the model
    def initialize(self, opt):
        BaseModel.initialize(self, opt)  # Call base class initialization
        if opt.resize_or_crop != 'none' or not opt.isTrain:
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc
        self.count = 0

    def gen_noise(self, shape):
        # Initialize a numpy array filled with zeros with the given shape
        noise = np.zeros(shape, dtype=np.uint8)

        # Generate random noise values
        noise = cv2.randn(noise, 0, 255)

        # Normalize the noise to range [0, 1]
        noise = np.asarray(noise / 255, dtype=np.uint8)

        # Convert the noise to a tensor and move it to GPU
        noise = torch.tensor(noise, dtype=torch.float32)

        return noise.cuda()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.autograd import Variable
    import cv2
    import numpy as np

    class ImageGenerator(nn.Module):

        def __init__(self, opt):
            super(ImageGenerator, self).__init__()
            self.opt = opt
            # Initialize your models and layers here, like G, G1, G2, Unet, etc.

        def multi_scale_blend(self, fake_img, fake_c, mask, number=4):
            """
            Blends two images (fake_img and fake_c) using a mask at multiple scales.
            """
            alpha = [0, 0.1, 0.3, 0.6, 0.9]
            smaller = mask
            out = 0
            for i in range(1, number + 1):
                bigger = smaller
                smaller = morpho(smaller, 2, False)
                mid = bigger - smaller
                out += mid * (alpha[i] * fake_c + (1 - alpha[i]) * fake_img)
            out += smaller * fake_c
            out += (1 - mask) * fake_img
            return out

        def forward(self, label, pre_clothes_mask, img_fore, clothes_mask, clothes, all_clothes_label, real_image, pose,
                    grid, mask_fore):
            """
            Main forward pass for the model that processes all input components and generates the output.
            """
            self.G1.eval()
            self.G.eval()
            self.Unet.eval()
            self.G2.eval()

            # Encode Inputs
            input_label, masked_label, all_clothes_label = self.encode_input(label, clothes_mask, all_clothes_label)
            arm1_mask = torch.FloatTensor((label.cpu().numpy() == 11).astype(float)).cuda()
            arm2_mask = torch.FloatTensor((label.cpu().numpy() == 13).astype(float)).cuda()
            pre_clothes_mask = torch.FloatTensor((pre_clothes_mask.detach().cpu().numpy() > 0.5).astype(float)).cuda()
            clothes = clothes * pre_clothes_mask

            shape = pre_clothes_mask.shape

            G1_in = torch.cat([pre_clothes_mask, clothes, all_clothes_label, pose, self.gen_noise(shape)], dim=1)
            arm_label = self.G1.refine(G1_in)

            arm_label = self.sigmoid(arm_label)
            CE_loss = self.cross_entropy2d(arm_label, (label * (1 - clothes_mask)).transpose(0, 1)[0].long()) * 10

            armlabel_map = generate_discrete_label(arm_label.detach(), 14, False)
            dis_label = generate_discrete_label(arm_label.detach(), 14)
            G2_in = torch.cat([pre_clothes_mask, clothes, dis_label, pose, self.gen_noise(shape)], 1)
            fake_cl = self.G2.refine(G2_in)
            fake_cl = self.sigmoid(fake_cl)
            CE_loss += self.BCE(fake_cl, clothes_mask) * 10

            fake_cl_dis = torch.FloatTensor((fake_cl.detach().cpu().numpy() > 0.5).astype(float)).cuda()
            fake_cl_dis = morpho(fake_cl_dis, 1, True)

            # Arm mask operations
            new_arm1_mask = torch.FloatTensor((armlabel_map.cpu().numpy() == 11).astype(float)).cuda()
            new_arm2_mask = torch.FloatTensor((armlabel_map.cpu().numpy() == 13).astype(float)).cuda()
            fake_cl_dis = fake_cl_dis * (1 - new_arm1_mask) * (1 - new_arm2_mask)
            fake_cl_dis *= mask_fore

            arm1_occ = clothes_mask * new_arm1_mask
            arm2_occ = clothes_mask * new_arm2_mask
            bigger_arm1_occ = morpho(arm1_occ, 10)
            bigger_arm2_occ = morpho(arm2_occ, 10)

            arm1_full = arm1_occ + (1 - clothes_mask) * arm1_mask
            arm2_full = arm2_occ + (1 - clothes_mask) * arm2_mask

            armlabel_map *= (1 - new_arm1_mask)
            armlabel_map *= (1 - new_arm2_mask)
            armlabel_map = armlabel_map * (1 - arm1_full) + arm1_full * 11
            armlabel_map = armlabel_map * (1 - arm2_full) + arm2_full * 13
            armlabel_map *= (1 - fake_cl_dis)
            dis_label = encode(armlabel_map, armlabel_map.shape)

            fake_c, warped, warped_mask, warped_grid = self.Unet(clothes, fake_cl_dis, pre_clothes_mask, grid)
            mask = fake_c[:, 3, :, :]
            mask = self.sigmoid(mask) * fake_cl_dis
            fake_c = self.tanh(fake_c[:, 0:3, :, :])
            fake_c = fake_c * (1 - mask) + mask * warped

            skin_color = self.ger_average_color((arm1_mask + arm2_mask - arm2_mask * arm1_mask),
                                                (arm1_mask + arm2_mask - arm2_mask * arm1_mask) * real_image)

            occlude = (1 - bigger_arm1_occ * (arm2_mask + arm1_mask + clothes_mask)) * (
                        1 - bigger_arm2_occ * (arm2_mask + arm1_mask + clothes_mask))
            img_hole_hand = img_fore * (1 - clothes_mask) * occlude * (1 - fake_cl_dis)

            G_in = torch.cat([img_hole_hand, dis_label, fake_c, skin_color, self.gen_noise(shape)], 1)
            fake_image = self.G.refine(G_in.detach())
            fake_image = self.tanh(fake_image)

            # Calculate losses
            loss_D_fake = 0
            loss_D_real = 0
            loss_G_GAN = 0
            loss_G_VGG = 0

            L1_loss = 0
            style_loss = L1_loss

            return [self.loss_filter(loss_G_GAN, 0, loss_G_VGG, loss_D_real, loss_D_fake), fake_image,
                    clothes, arm_label, L1_loss, style_loss, fake_cl, CE_loss, real_image, warped_grid]

        def inference(self, label, label_ref, image_ref):
            """
            Inference method that generates images from given labels and reference images.
            """
            image_ref = Variable(image_ref)
            input_label, input_label_ref, real_image_ref = self.encode_input_test(Variable(label), Variable(label_ref),
                                                                                  image_ref, infer=True)

            if torch.__version__.startswith('0.4'):
                with torch.no_grad():
                    fake_image = self.netG.forward(input_label, input_label_ref, real_image_ref)
            else:
                fake_image = self.netG.forward(input_label, input_label_ref, real_image_ref)

            return fake_image

        def save(self, which_epoch):
            """
            Method to save the model's networks and parameters after training.
            """
            # Saving models is currently not implemented, but you would typically store them like so:
            # self.save_network(self.Unet, 'U', which_epoch, self.gpu_ids)
            # self.save_network(self.G, 'G', which_epoch, self.gpu_ids)
            # self.save_network(self.G1, 'G1', which_epoch, self.gpu_ids)
            # self.save_network(self.G2, 'G2', which_epoch, self.gpu_ids)
            pass

        def update_fixed_params(self):
            """
            Method to update and fine-tune fixed parameters after a set number of iterations.
            """
            params = list(self.netG.parameters())
            if self.gen_features:
                params += list(self.netE.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            if self.opt.verbose:
                print('------------ Now also finetuning global generator -----------')

        def update_learning_rate(self):
            """
            Method to decay and update the learning rate.
            """
            lrd = self.opt.lr / self.opt.niter_decay
            lr = self.old_lr - lrd
            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = lr
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = lr
            if self.opt.verbose:
                print('update learning rate: %f -> %f' % (self.old_lr, lr))
            self.old_lr = lr

    class InferenceModel(Pix2PixHDModel):
        def forward(self, inp):
            """
            Forward pass for inference only, using the trained model.
            """
            label = inp
            return self.inference(label)
