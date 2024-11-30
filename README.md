## Towards Photo-Realistic Virtual Try-On by Adaptively Generating↔Preserving Image Content, 


**Notably, virtual try-on is a difficult research topic, and our solution is of course not perfect. 

The code is not fully tested. If you meet any bugs or want to improve the system,


[[Sample Try-on Video]](https://www.youtube.com/watch?v=BbKBSfDBcxI) [[Checkpoints]](https://drive.google.com/file/d/1UWT6esQIU_d4tUm8cjxDKMhB8joQbrFx/view?usp=sharing) 

[[Dataset_Test]](https://drive.google.com/file/d/1tE7hcVFm8Td8kRh5iYRBSDFdvZIkbUIR/view?usp=sharing) [Dataset_Train](https://drive.google.com/file/d/1lHNujZIq6KVeGOOdwnOXVCSR5E7Kv6xv/view?usp=sharing)


[[Paper]](https://arxiv.org/abs/2003.05863)


Download the dataset_test folder from the Google Drive link.
Place it inside the Data_preprocessing folder of your project
ACGPN_inference/
├── Data_preprocessing/
│   ├── dataset_test/

Download the Checkpoints folder from the provided link.
Place it inside the ACGPN_inference/checkpoints/label2city folder
ACGPN_inference/
├── checkpoints/
│   ├── label2city/

The result is provided in this link.[https://gofile.io/d/TuQl5A]

## Inference

- Setup the env befor testing (windows):
```bash
python3 -m venv venv
.\.venv\Scripts\activate
cd ACGPN_inference
pip install -r requirements.txt
```
- Check the cuda:
```bash
python cuda_check.py
```
## Evaluation IS and SSIM
**Note that** The released checkpoints are different from what we used in the paper which generate better visual results but may have different (lower or higher) quantitative statistics. Same results of the paper can be reproduced by re-training with different training epochs.

The results for computing IS and SSIM are **same-clothes reconstructed results**. 

The code *defaultly* generates *random* clothes-model pairs, so you need to modify **ACGPN_inference/data/aligned_dataset.py** to generate the reconstructed results.

Here, we also offer the reconstructed results on test set of VITON dataset by inferencing this github repo, 
[[Precomputed Evaluation Results]](https://drive.google.com/file/d/1obk8NFMlSFmCJJuzJDooSWesI46ZXXmY/view?usp=sharing)
The results here can be directly used to compute the IS and SSIM evalutations. You can get identical results using this github repo.



### SSIM score
  1. Use the pytorch SSIM repo. https://github.com/Po-Hsun-Su/pytorch-ssim
  2. Normalize the image (img/255.0) and reshape correctly. If not normalized correctly, the results differ a lot. 
  3. Compute the score with window size = 11. The SSIM score should be 0.8664, which is a higher score than reported in paper since it is a better checkpoint.


### IS score
  1. Use the pytorch inception score repo. https://github.com/sbarratt/inception-score-pytorch
  2. Normalize the images ((img/255.0)*2-1) and reshape correctly. Please strictly follow the procedure given in this repo.
  3. Compute the score. The splits number also changes the results. We use splits number =1 to compute the results.
  4. **Note that** the released checkpoints produce IS score 2.82, which is **slightly** lower (but still **SOTA**) than the paper since it is a different checkpoint with better SSIM performance. 


## The specific key points we choose to evaluate the try-on difficulty
![image](https://github.com/switchablenorms/DeepFashion_Try_On/blob/master/images/criterion.png)

- We use the pose map to calculate the difficulty level of try-on. The key motivation behind this is the more complex the occlusions and layouts are in the clothing area, the harder it will be. And the formula is given below. Also, manual selection is involved to improve the difficulty partition.
- Variations of the pose map predictions largely affect the absolute value of try-on complexity, so you may have different partition size using our reported separation values. 
- Relative ranking of complexity best depicts the complexity distribution. Try top 100 or bottom 100 and you can see the effectiveness of our criterion.
## The formula to compute the difficulty of try-on reference image

![image](https://github.com/switchablenorms/DeepFashion_Try_On/blob/master/images/formula.png)

where t is a certain key point, Mp' is the set of key point we take into consideration, and N is the size of the set. 
## Segmentation Label
```bash
0 -> Background
1 -> Hair
4 -> Upclothes
5 -> Left-shoe 
6 -> Right-shoe
7 -> Noise
8 -> Pants
9 -> Left_leg
10 -> Right_leg
11 -> Left_arm
12 -> Face
13 -> Right_arm
```
## Sample images from different difficulty level

![image](https://github.com/switchablenorms/DeepFashion_Try_On/blob/master/images/difficulty.png)

