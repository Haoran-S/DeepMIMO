## DeepMIMO for Power Allocation

This is a MATLAB / Python code package modified from DeepMIMO (https://www.deepmimo.net) to generate real channels for power allocation. This code package is related to the following article:

[1] Haoran Sun, Wenqiang Pu, Minghe Zhu,  Xiao Fu, Tsung-Hui Chang, and Mingyi Hong, "Learning to Continuously Optimize Wireless Resource In Episodically Dynamic Environment." arXiv preprint arXiv:2011.07782 (2020).
 
[2] Ahmed Alkhateeb, “DeepMIMO: A Generic Deep Learning Dataset for Millimeter Wave and Massive MIMO Applications,” in Proc. of Information Theory and Applications Workshop (ITA), San Diego, CA, Feb. 2019.
  
[3] Haoran Sun, Xiangyi Chen, Qingjiang Shi, Mingyi Hong, Xiao Fu and Nikos D. Sidiropoulos, “Learning to Optimize: Training Deep Neural Networks for Wireless Resource Management”, IEEE Transactions on Signal Processing 66.20 (2018): 5438-5453. 
 
 
## Dataset Generation
To generate the dataset, please follow these steps:

Step 1: Download source data 'O1_60' under ‘O1’ Ray-Tracing Scenario from https://www.deepmimo.net/ray_tracing.html

Step 2: Run Matlab file generate_data_part1.m to generate wireless channels based on DeepMIMO ray-tracing scenario

Step 3: Run Python file python3 generate_data_part2.py to generate labels labeled by WMMSE algorithm
