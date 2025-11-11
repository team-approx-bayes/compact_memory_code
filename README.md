
## Compact Memory for Continual Logistic Regression

This is an implemenation for "Compact Memory for Continual Logistic Regression"



## Description for implementation

* example_four_moons.ipynb                 : results for four-moon task

* setting_dataset.py                       : train R-FVI with informative priors for classification
* main_splitcifar100_basereplay_batch.py   : baseline experience replay for Split-CIFAR-100
* main_splitcifar100_baselambda_batch.py   : baseline K-prior  for Split-CIFAR-100
* main_splitcifar100_ourem_batch.py        : our method for Split-CIFAR-100 

* run_main_splitcifar100.sh                : execute experimentsr for Split-CIFAR-100  


* Once you replace generate_setting_splitcifar100 in each main_**.py using another dataset generator in setting_dataset.py, the code can be run and evaluated on other datasets as well.

