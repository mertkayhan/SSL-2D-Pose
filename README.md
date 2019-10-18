# Deep Attention Based Semi-Supervised 2D-Pose Estimation
Code for the paper "Deep Attention Based Semi-Supervised 2D-Pose Estimation for Surgical Instruments" is presented in this repository.

<div align="center" style="width:image width px;">
  <img  src="https://github.com/mertkayhan/SSL-2D-Pose/blob/master/README/Endovis.gif" width=500 alt="Endovis results">
  <img  src="https://github.com/mertkayhan/SSL-2D-Pose/blob/master/README/RMIT.gif" width=500 alt="RMIT results">
</div>


Figure: Example demonstration of the test set results can be seen in the above provided videos. (Left: Endovis dataset, Right: RMIT dataset)

## Abstract

For many practical problems and applications, it is not feasible to create a vast and accurately labeled dataset, which restricts the application of deep learning in many areas. Semi-supervised learning algorithms intend to improve performance by also leveraging unlabeled data. This is very valuable for 2D-pose estimation task where data labeling requires substantial time and is subject to noise. This work aims to investigate if semi-supervised learning techniques can achieve acceptable performance level that makes using these algorithms during training justifiable. To this end, a lightweight network architecture is introduced and mean teacher, virtual adversarial training and pseudo-labeling algorithms are evaluated on 2D-pose estimation for surgical instruments. For the applicability of pseudo-labelling algorithm, we have proposed a novel confidence measure, total variation. Experimental results show that utilization of semi-supervised learning improves the performance on unseen geometries drastically while maintaining high accuracy for seen geometries. For RMIT benchmark, our lightweight architecture outperforms state-of-the-art with supervised learning. For Endovis benchmark, pseudo-labelling algorithm improves the supervised baseline achieving the new state-of-the-art performance.

## Running the Code

* Scripts that were used to generate target heatmaps are provided in the [utils](https://github.com/mertkayhan/SSL-2D-Pose/tree/master/utils) folder.
* For RMIT dataset, data is expected to be in `$ROOT/training/image` and `$ROOT/test/image` respectively.
* For RMIT dataset, labels are expected to be in `$ROOT/training/label` and `$ROOT/test/label` respectively.
* For EndoVis dataset, data is expected to be in `$ROOT/labelled_train` and `$ROOT/labelled_test` respectively.
* For EndoVis dataset, labels are expected to be in `$ROOT/pseudo_labels`, `$ROOT/training_labels` and `$ROOT/test_labels` respectively.
* Models can be evaluated using [evaluate.py](https://github.com/mertkayhan/SSL-2D-Pose/blob/master/evaluate.py) after training. (Post-processing labels are relevant for the evaluation script and these labels should be saved in `$ROOT/training_labels_postprocessing` and `$ROOT/test_labels_postprocessing` respectively.)
* Training can be performed using [train.py](https://github.com/mertkayhan/SSL-2D-Pose/blob/master/train.py).

`python train.py --batch_size 5 --gpu_id 0 --root <data-folder> --use_vat <toggle-vat> --use_mean_teacher <toggle-mean-teacher> --use_pseudo_labels <pseudo-labeling> --dataset <RMIT|ENDOVIS>`

## Citation

Please cite the following article if you use this code: 
<b>TODO


