# PR
This is the project for Pattern Recognition course.
Platform:
- ubuntu 16.04
- opencv 3.1.0
- tensorflow (forgot the version...)
- pascal voc 2007
- imagenet-vgg-verydeep-19.mat
- and more (add later...)

## Image-fine-tuning with trick
some paramters should be checked out before the scripts

```sh
# generate the datasets
python generate_datasets.py

# generate the train and test list
python generate_train_and_test_lists.py

# start the fine-tuning process
# you need imagenet-vgg-verydeep-19.mat in data/ folder
python finetune_vgg_19.py
```

## Hypotheses-fine-tuning (in the dirctory H-FT/)
some paramters should be checked out before the scripts

```sh
# generate the hypotheses
python hypotheses.py

# start the fine-tuning process
python final.py
```
