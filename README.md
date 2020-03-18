

# IJCAI2019_DDNN
Code for our IJCAI 2019 paper "A Novel Distribution-Embedded Neural Network for Sensor-Based Activity Recognition". If you find the codes helpful, kindly cite our paper. 


> ```
>@inproceedings{DBLP:conf/ijcai/QianPDM19,
>  author    = {Hangwei Qian and
>               Sinno Jialin Pan and
>               Bingshui Da and
>               Chunyan Miao},
>  title     = {A Novel Distribution-Embedded Neural Network for Sensor-Based Activity
>               Recognition},
>  booktitle = {{IJCAI}},
>  pages     = {5614--5620},
>  publisher = {ijcai.org},
>  year      = {2019}
>}
> ```


The codes are tested with pytorch=1.3, python>3.5. Codes for data preprocessing on 4 datasets (Opportunity, UCIHAR, DG and PAMAP2) are in files `data_preprocess_[dataset_name].py`. To run the code, please run the following script:

```
python main_[dataset_name].py 
```




