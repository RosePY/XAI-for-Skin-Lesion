# XAI-for-Skin-Lesion
Repository for Are Explanations Helpful? A Comparative Analysis of Explainability Methods in Skin Lesion Classifiers paper

_Code and models to be added_

## Experimental Setup for Explainability Methods
All experiments were conducted on a single NVIDIA Quadro RTX 8000 GPU, with 48 GB of GDDR6 (Graphics Double Data Rate 6) synchronous dynamic RAM.

Since some methods work on PyTorch and others on TensorFlow, we converted the PyTorch models to TensorFlow using pytorch2keras (https://github.com/gmalivenko/pytorch2keras). Thus, we used the same learned weights and got the same prediction. In the following, we describe the hyperparameters used for each method.

- We chose the last convolutional layer of both models for Grad-CAM, Score-CAM, ACE, and ICE. The implementation used for Grad-CAM and Score-CAM is available on Github\footnote{\url{https://github.com/yiskw713/ScoreCAM}}.
    
- For LIME, we used the library implemented by the authors (https://github.com/marcotcr/lime), we used a Ridge Regression linear model, a cosine distance function, and an exponential kernel. The saliency feature set is created using the top 5 features (superpixels created with QuickShif) that positively impact the model's prediction.
    
- For SHAP, we used Kernel Shap from the author's implementation (https://github.com/slundberg/shap) with the same superpixels used on LIME.
   
- For ACE, we followed the authors, we selected a random set of 50 images in the melanoma class, and to represent the random concept in the statistical significance test, we chose 50 images of the whole ISIC 2018 dataset; likewise, we chose 50 random images for each of the 50 random sets. We performed a SLIC (Simple Linear Iterative Clustering) superpixel segmentation with 15, 50, and 80 segments to get the concepts' patches. These segments are completed with a gray value of 117.5 and passed through the networks to get their representation on the last layer. We used $k$-Means with $k=25$ to cluster the representations and find the concepts. We removed clusters with few elements. For the TCAV score, the p-value is 0.05, so concepts with \textit{p-value} greater than 0.05 have not passed the statistical significance test. Since the original code (https://github.com/amiratag/ACE/) from the authors was on TensorFlow version 1, an upgraded version on Github (https://github.com/monz/ACE/tree/tensorflow-2-upgrade) was used and modified to be run with the selected~models.
    
- For ICE, we used the implementation on Github (https://github.com/zhangrh93/InvertibleCE) provided by the authors, we chose randomly 78 images per class, NMF is trained with a limit of 200 iterations, 16 components and 64 as batch size.
- For CME, we used the code implementation on Githhub (https://github.com/dmitrykazhdan/CME) we chose the last 5 layers from which we learn concepts. Logistic regression for input to a concept is trained with a maximum of 200 iterations. We used Minimal Cost-Complexity Pruning for the Decision Tree with an alpha value of 0.00333.

