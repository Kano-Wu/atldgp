# ATL-DGP

## Introdunction 

### Paper

M.Kandemir, Asymmetric Transfer Learning with Deep Gaussian Processes, ICML, 2015.

Contact: melihkandemir@gmail.com

> From Github: https://github.com/melihkandemir/atldgp

### Source Code

* DSGPSymmetricTransferClassifier.py
* DSGPAsymmetricTransferClassifier.py
* Kernel.py
* LinearKernel.py
* RBFKernel.py
* MultiClassPrediction.py
* Prediction.py
* demo.py

### Data Format

```
Xtrain_source： (sTrain_Num , Dim)
ytrain_source： (sTrain_Num ,  1 )

Xtrain_target： (tTrain_Num , Dim)
ytrain_target：	 (tTrain_Num ,  1 )

Data： (sTrain_Num + tTrain_Num , Dim)
labels： (sTrain_Num + tTrain_Num , 1 )

Xtest_target：	 (tTest_Num , Dim)
ytest_target：	 (tTest_Num , 1  ) 
```

### Run Example

```
> python demo.py
```

## For Sentiment Analysis

### More Code

* DSGPAsymmetricTransferClassifier_mod.py
* my_demo.py -- main
* mapped.py
* tmp.py





