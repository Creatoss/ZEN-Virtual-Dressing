import time
from collections import OrderedDict
from options.train_options import TrainOptions # Import options for training configurations
from data.data_loader import CreateDataLoader # Import functionality for creating the data loader
from models.models import create_model # Import functionality for creating the model
import util.util as util # Import utility functions
import os
import numpy as np
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter # Import TensorBoard for logging
import cv2

writer = SummaryWriter('runs/G1G2') # Set up a directory for TensorBoard logs
SIZE=320 # Define size constant
NC=14 # Number of channels/classes

torch.cuda.empty_cache() # Clear GPU memory cache

def generate_label_plain(inputs):
    """
    Generate plain labels by taking the max of the input tensor along the channel axis.
    Inputs:
    - inputs: Tensor of size [batch_size, NC, height, width]
    """
    size = inputs.size()
    pred_batch = []
    for input in inputs:
        input = input.view(1, NC, 256,192) # Reshape input to a specific size
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0) # Extract the max index for each spatial location
        pred_batch.append(pred)

    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch) # Convert to tensor
    label_batch = pred_batch.view(size[0], 1, 256,192) # Reshape the labels to match the batch size

    return label_batch

def generate_label_color(inputs):
    """
    Generate color-coded labels from the input tensor using a utility function.
    Inputs:
    - inputs: List of tensors
    """
    label_batch = []
    for i in range(len(inputs)):
        label_batch.append(util.tensor2label(inputs[i], opt.label_nc)) # Convert tensor to label using a predefined function
    label_batch = np.array(label_batch)
    label_batch = label_batch * 2 - 1 # Normalize labels
    input_label = torch.from_numpy(label_batch)

    return input_label

def complete_compose(img, mask, label):
    """
    Compose an image using the mask and label.
    Inputs:
    - img: Input image tensor
    - mask: Binary mask tensor
    - label: Label tensor
    """
    label = label.cpu().numpy() # Convert label tensor to numpy array
    M_f = label > 0 # Generate binary mask based on label
    M_f = M_f.astype(np.int) # Convert mask to integer
    M_f = torch.FloatTensor(M_f).cuda() # Move mask to GPU
    masked_img = img * (1 - mask) # Apply mask to the image
    M_c = (1 - mask.cuda()) * M_f # Combine mask and label information
    M_c = M_c + torch.zeros(img.shape).cuda() # Broadcast to match the image shape
    return masked_img, M_c, M_f

def compose(label, mask, color_mask, edge, color, noise):
    """
    Compose multiple components into a single output.
    Inputs:
    - label: Label tensor
    - mask: Mask tensor
    - color_mask: Color mask tensor
    - edge: Edge tensor
    - color: Color tensor
    - noise: Noise tensor
    """
    masked_label = label * (1 - mask) # Mask the label
    masked_edge = mask * edge # Mask the edges
    masked_color_strokes = mask * (1 - color_mask) * color # Mask the color strokes
    masked_noise = mask * noise # Mask the noise
    return masked_label, masked_edge, masked_color_strokes, masked_noise

def changearm(old_label):
    """
    Modify label tensor to reassign specific regions.
    Inputs:
    - old_label: Input label tensor
    """
    label = old_label
    arm1 = torch.FloatTensor((data['label'].cpu().numpy() == 11).astype(int)) # Identify region with label 11
    arm2 = torch.FloatTensor((data['label'].cpu().numpy() == 13).astype(int)) # Identify region with label 13
    noise = torch.FloatTensor((data['label'].cpu().numpy() == 7).astype(int)) # Identify noise regions
    label = label * (1 - arm1) + arm1 * 4 # Reassign label for arm1
    label = label * (1 - arm2) + arm2 * 4 # Reassign label for arm2
    label = label * (1 - noise) + noise * 4 # Reassign label for noise
    return label

os.makedirs('sample', exist_ok=True) # Create directory for saving samples if it doesn't exist
opt = TrainOptions().parse() # Parse training options
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt') # Path to save iteration information

if opt.continue_train:
    # Load previous training state if resuming
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
else:
    start_epoch, epoch_iter = 1, 0

if opt.debug:
    # Adjust options for debugging
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

data_loader = CreateDataLoader(opt) # Initialize data loader
dataset = data_loader.load_data() # Load dataset
dataset_size = len(data_loader) # Get dataset size
print('# Inference images = %d' % dataset_size)

model = create_model(opt) # Create the model

total_steps = (start_epoch - 1) * dataset_size + epoch_iter # Calculate total training steps

display_delta = total_steps % opt.display_freq # Calculate display interval
print_delta = total_steps % opt.print_freq # Calculate print interval
save_delta = total_steps % opt.save_latest_freq # Calculate save interval

step = 0 # Initialize step counter


for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1): # Loop over epochs
    epoch_start_time = time.time() # Record the start time of the epoch
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size # Reset iteration counter for the dataset
    for i, data in enumerate(dataset, start=epoch_iter): # Loop over batches in the dataset

        iter_start_time = time.time() # Record the start time of the iteration
        total_steps += opt.batchSize # Increment the total steps by the batch size
        epoch_iter += opt.batchSize # Increment the epoch iteration by the batch size

        # whether to collect output images
        save_fake = True # Always save fake outputs for visualization

        ## add gaussian noise channel
        ## wash the label
        t_mask = torch.FloatTensor((data['label'].cpu().numpy() == 7).astype(float)) # Generate a binary mask for label 7
        #
        # Modify the labels for processing
        mask_clothes = torch.FloatTensor((data['label'].cpu().numpy() == 4).astype(int)) # Mask regions with label 4 (clothes)
        mask_fore = torch.FloatTensor((data['label'].cpu().numpy() > 0).astype(int)) # Mask foreground regions
        img_fore = data['image'] * mask_fore # Extract the foreground image
        img_fore_wc = img_fore * mask_fore # Weighted foreground image
        all_clothes_label = changearm(data['label']) # Modify arm regions in the label

        ############## Forward Pass ######################
        # Perform forward pass to compute losses and outputs
        losses, fake_image, real_image, input_label, L1_loss, style_loss, clothes_mask, CE_loss, rgb, alpha = model(
            Variable(data['label'].cuda()),
            Variable(data['edge'].cuda()),
            Variable(img_fore.cuda()),
            Variable(mask_clothes.cuda()),
            Variable(data['color'].cuda()),
            Variable(all_clothes_label.cuda()),
            Variable(data['image'].cuda()),
            Variable(data['pose'].cuda()),
            Variable(data['image'].cuda()),
            Variable(mask_fore.cuda())
        )

        # Sum losses across devices
        losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
        loss_dict = dict(zip(model.module.loss_names, losses)) # Map losses to their names

        # Calculate generator and discriminator losses
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5 # Average discriminator loss
        loss_G = loss_dict['G_GAN'] + torch.mean(CE_loss) # Generator loss with cross-entropy

        # Log losses to TensorBoard
        writer.add_scalar('loss_d', loss_D, step)
        writer.add_scalar('loss_g', loss_G, step)
        writer.add_scalar('loss_CE', torch.mean(CE_loss), step)
        writer.add_scalar('loss_g_gan', loss_dict['G_GAN'], step)

        ############### Backward Pass ####################
        # Backpropagation and weight updates (commented out in current code)
        # Update generator weights
        # model.module.optimizer_G.zero_grad()
        # loss_G.backward()
        # model.module.optimizer_G.step()
        #
        # Update discriminator weights
        # model.module.optimizer_D.zero_grad()
        # loss_D.backward()
        # model.module.optimizer_D.step()

        ############## Display results and errors ##########
        ### Display output images
        a = generate_label_color(generate_label_plain(input_label)).float().cuda() # Generate colored labels
        b = real_image.float().cuda() # Real image
        c = fake_image.float().cuda() # Fake image
        d = torch.cat([clothes_mask, clothes_mask, clothes_mask], 1) # Create 3-channel mask
        combine = torch.cat([a[0], d[0], b[0], c[0], rgb[0]], 2).squeeze() # Combine multiple outputs for visualization
        cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2 # Convert tensor to image format
        if step % 1 == 0: # Save images at every step
            writer.add_image('combine', (combine.data + 1) / 2.0, step) # Log combined image to TensorBoard
            rgb = (cv_img * 255).astype(np.uint8) # Scale image for saving
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) # Convert RGB to BGR for OpenCV
            n = str(step) + '.jpg' # Generate image name
            cv2.imwrite('sample/' + data['name'][0], bgr) # Save image to disk
        step += 1 # Increment step counter
        print(step) # Print step

        ### Save latest model
        if total_steps % opt.save_latest_freq == save_delta: # Save model at specified intervals
            # Uncomment to enable saving
            # print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            # model.module.save('latest')
            # np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
            pass
        if epoch_iter >= dataset_size: # End epoch when all data is processed
            break

    # end of epoch
    iter_end_time = time.time() # Record end time of the epoch
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    break # Break after one epoch (for debugging purposes)

    ### Save model for this epoch
    if epoch % opt.save_epoch_freq == 0: # Save model at regular intervals
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        model.module.save('latest')
        model.module.save(epoch)
        # np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    ### Train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        model.module.update_fixed_params()

    ### Linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.module.update_learning_rate()

