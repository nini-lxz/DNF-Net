# DNF-Net: a Deep Normal Filtering Network for Mesh Denoising
by [Xianzhi Li](https://nini-lxz.github.io/), [Ruihui Li](https://liruihui.github.io/), [Lei Zhu](https://appsrv.cse.cuhk.edu.hk/~lzhu/), [Chi-Wing Fu](https://www.cse.cuhk.edu.hk/~cwfu/), and [Pheng-Ann Heng](http://www.cse.cuhk.edu.hk/~pheng/).

### Introduction
This repository is for our IEEE Transactions on Visualization and Computer Graphics (TVCG) 2020 paper 'DNF-Net: a Deep Normal Filtering Network for Mesh Denoising'. In this paper, we present a deep normal filtering network, called DNF-Net, for mesh denoising. To better capture local geometry, our network processes the mesh in terms of local patches extracted from the mesh. Overall, DNF-Net is an end-to-end network that takes patches of facet normals as inputs and directly outputs the corresponding denoised facet normals of patches. In this way, we can reconstruct the geometry from the denoised normals with feature preservation. Besides the overall network architecture, our contributions include a novel multi-scale feature embedding unit, a residual learning strategy to remove noise, and a deeply-supervised joint loss function.

### Usage
(1) You can directly test your noisy meshes using our previously-trained network:
Step1: First, since our DNF-Net takes patches as network inputs, so you need to cut the noisy mesh into patches. You can download the C++ code [here](https://drive.google.com/file/d/1b3XOnDw-8zuw-QII96qYUh6aHSYuvEvS/view?usp=sharing). Unzip the code folder, inside the `Meshviewer` folder, open `meshviewer.cpp`, and you will see the main function. In the `prepareTestData()` function, you need to specify the path and name of your noisy meshes. Then the function will generate patches per face automatically.
Step2: zip these patches into h5 file. The python code is provided in the `code` folder --> `patch_to_h5.py`.
Step3: feed the h5 file to our trained network for testing. The python code is provided in the `code` folder --> `test.py`. After running, the network will output a txt file. In the txt file, the three numbers in each line indicate the denoised facet normal of each face.
Step4: reconstruct the denoised mesh from the denoised facet normals. Still use the C++ code downloaded in Step1. In the main function, there is another function called `generateDenoisedMesh()`. You need to specify the noisy mesh and the vertex iteration number. You can also give the path of ground-truth mesh to calculate the denoising error.  

### Questions
Please constact 'lixianzhi123@gmail.com'

