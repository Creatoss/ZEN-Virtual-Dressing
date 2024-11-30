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

## Sample Try-on Results
  
![image](https://github.com/switchablenorms/DeepFashion_Try_On/blob/master/images/tryon.png)

## Limitations and Failure Cases
![image](https://github.com/switchablenorms/DeepFashion_Try_On/blob/master/images/failure.png)
	1. Large transformation of the semantic layout is hard to handle, partly ascribing to the agnostic input of fused segmentation.
	2. The shape of the original clues is not completely removed. The same problem as VITON.
	3. Very difficult pose is hard to handle. Better solution could be proposed.

## Training Details
Due to some version differences of the code, and some updates for better quality, some implementation details may be different from the paper. 

For better inference performance, model G and G2 should be trained with 200 epoches, while model G1 and U net should be trained with 20 epoches.

## License
The use of this software is RESTRICTED to **non-commercial research and educational purposes**.

## Citation
If you use our code or models or the offered baseline results in your research, please cite with:
```
@InProceedings{Yang_2020_CVPR,
author = {Yang, Han and Zhang, Ruimao and Guo, Xiaobao and Liu, Wei and Zuo, Wangmeng and Luo, Ping},
title = {Towards Photo-Realistic Virtual Try-On by Adaptively Generating-Preserving Image Content},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}

@inproceedings{ge2021disentangled,
  title={Disentangled Cycle Consistency for Highly-realistic Virtual Try-On},
  author={Ge, Chongjian and Song, Yibing and Ge, Yuying and Yang, Han and Liu, Wei and Luo, Ping},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16928--16937},
  year={2021}
}

@inproceedings{yang2022full,
title = {Full-Range Virtual Try-On With Recurrent Tri-Level Transform},
author = {Yang, Han and Yu, Xinrui and Liu, Ziwei},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
pages = {3460--3469}
year = {2022}
}

```

## Dataset
**VITON Dataset** This dataset is presented in [VITON](https://github.com/xthan/VITON), containing 19,000 image pairs, each of which includes a front-view woman image and a top clothing image. After removing the invalid image pairs, it yields 16,253 pairs, further splitting into a training set of 14,221 paris and a testing set of 2,032 pairs.