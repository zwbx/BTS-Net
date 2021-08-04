# BTS-Net (ICME 2021)
BTS-NET: BI-DIRECTIONAL TRANSFER-AND-SELECTION NETWORK FOR RGB-D SALIENT OBJECT DETECTION(https://arxiv.org/abs/2104.01784)

<p align="center">
    <img src="Img/Diagram.png" width="80%"/> <br />
 <em> 
     Block diagram of the proposed BTS-Net.
    </em>
</p>

## 1. Requirements

 - Python 3.7, Pytorch 1.7, Cuda 10.1
 - Test on Win10 and Ubuntu 16.04

## 2. Data Preparation

 - Download the raw data from [Here](https://pan.baidu.com/s/1fTjhNJydzZpMkdlhwdFkAw) [code: 5x0o] and trained model (epoch_100.pth) from [Here](https://pan.baidu.com/s/1SbNnFmeW5vHj6tFWgQLSpg) 
[code: 2j99]. Then put them under the following directory:
 
        -dataset\ 
          -NJU2K\  
          -NLPR\
          ...
        -pretrain
          -epoch_100.pth
          ...

	  
## 3. Testing

    Directly run test.py
    
    The test maps will be saved to './resutls/'.

- Evaluate the result maps:
    
    You can evaluate the result maps using the tool in [Matlab Version](http://dpfan.net/d3netbenchmark/) or [Python_GPU Version](https://github.com/zyjwuyan/SOD_Evaluation_Metrics).
    
- If you need the train code, please send to the email (zhangwenbo@scu.stu.edu.cn). 

## 4. Results

</p>
<p align="center">
    <img src="./Img/comparison.png" width="100%"/> <br />
 <em>
  Quantitative comparison with 16 SOTA over 4 metrics (S-measure, max F-measure, max E-measure and MAE) on 6 datasets. 
  </em>
</p>

### 4.3 Download
 - Test results of the above datasets can be download from [here](https://pan.baidu.com/s/1MGKCsNW8qw40_PJ0RCyAtA) [code: xo9p].
 
## 5. Citation

Please cite the following paper if you use this repository in your reseach

	@inproceedings{Zhang2021BTSNet,
 	 title={BTS-Net: Bi-directional Transfer-and-Selection Network for RGB-D Salient Object Detection},
	  author={Wenbo Zhang and Yao Jiang and Keren Fu and Qijun Zhao},
	  booktitle={ICME},
	  year={2021}
	}

