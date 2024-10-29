# DNF-Net: a Deep Normal Filtering Network for Mesh Denoising
by [Xianzhi Li](https://nini-lxz.github.io/), [Ruihui Li](https://liruihui.github.io/), [Lei Zhu](https://appsrv.cse.cuhk.edu.hk/~lzhu/), [Chi-Wing Fu](https://www.cse.cuhk.edu.hk/~cwfu/), and [Pheng-Ann Heng](http://www.cse.cuhk.edu.hk/~pheng/).

### Introduction
This repository is for our IEEE Transactions on Visualization and Computer Graphics (TVCG) 2020 paper '[DNF-Net: a Deep Normal Filtering Network for Mesh Denoising](https://arxiv.org/pdf/2006.15510.pdf)'. In this paper, we present a deep normal filtering network, called DNF-Net, for mesh denoising. To better capture local geometry, our network processes the mesh in terms of local patches extracted from the mesh. Overall, DNF-Net is an end-to-end network that takes patches of facet normals as inputs and directly outputs the corresponding denoised facet normals of patches. In this way, we can reconstruct the geometry from the denoised normals with feature preservation. Besides the overall network architecture, our contributions include a novel multi-scale feature embedding unit, a residual learning strategy to remove noise, and a deeply-supervised joint loss function.

### Usage

#### Environment Requirements
- Python 2.7
- Tensorflow 1.9.0
- Ubuntu 16.04.6
- CUDA 10.2

#### Try our previously-trained network:
(1) Step 1: prepare network input <br>
You need to cut the noisy mesh into patches. Please download the C++ code [here](https://pan.baidu.com/s/12RWgs_wo206ZI2DBjR5_Sw?pwd=brrb) with the download code "brrb". Unzip the file, inside the `Meshviewer` folder, open `meshviewer.cpp`, and you will see the main function. In the `prepareTestData()` function, you need to specify the path and name of your noisy meshes. Then the function will generate patches per face automatically. Finally, use the code `patch_to_h5.py` inside the `code` folder (see this GitHub page) to generate h5 file. <br>

(2) Step 2: testing to generate the denoised facet normals<br>
Feed the h5 file to our trained network for testing. The python code is provided in the `code` folder --> `test.py`. After running, the network will output a txt file. In the txt file, the three values in each line indicate the denoised facet normal of each face.

(3) Step 3: generate the denoised mesh <br>
Still use the C++ code downloaded in Step 1 to reconstruct the denoised mesh from the denoised facet normals. In the main function, there is a function called `generateDenoisedMesh()`. You need to specify the noisy mesh and the vertex iteration number. You can also give the path of ground-truth mesh to calculate the denoising error. For our provided three meshes, we have specified the corresponding recommended vertex iteration number.

#### Re-train our network:
If you want to re-train our network, please email me to ask for our synthetic training data.

### Questions
Please contact 'lixianzhi123@gmail.com'

