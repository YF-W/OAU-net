# OAU-net
 OAU-net: Outlined Attention U-net for Biomedical Image Segmentation

authors: Haojie Song , Yuefei Wang, Shijie Zeng , Xiaoyan Guo , Zheheng Li

In this paper, we propose an Outlined Attention U-network (OAU-net) with bypass branching strategy to solve biomedical image segmentation tasks, which is capable of sensing shallow and deep features. Unlike previous studies, we use residual convolution and res2convolution as encoders. In particular, the outline filter and attention module are embedded in the skip connection part, respectively. Shallow features will enhance the edge information after being processed by the outline filter. Meanwhile, in the depths of the network, to better realize feature fusion, our attention module will simultaneously emphasize the independence between feature map channels (channel attention module) and each position information (spatial attention module), that is, the hybrid domain attention module. Finally, we conducted ablation experiments and comparative experiments according 
to three public data sets (pulmonary CT lesions, Kaggle 2018 data science bowl, skin lesions), and analyzed them with classical evaluation indexes. Experimental results show that our proposed method improves segmentation accuracy effectively. 

![image](https://github.com/YF-W/OAU-net/assets/66008255/29134e0c-6304-4595-b06c-eb9bec172273)

please see https://www.sciencedirect.com/science/article/abs/pii/S1746809422005158
