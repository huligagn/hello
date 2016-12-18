# TPAMI
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
- generate the train and test list

```sh
python generate_train_and_test_lists.py
```

- generate the datasets according to the lists in step1

```sh
python generate_datasets.py
```

- start the fine-tuning process

```sh
python finetune_vgg_19.py
```

## Hypotheses-fine-tuning (in the dirctory src/)
some paramters should be checked out before the scripts
- generate the hypotheses

```sh
python hypotheses.py
```

- start the fine-tuning process

```sh
python final.py
```
