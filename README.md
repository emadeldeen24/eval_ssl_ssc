# Self-supervised Learning for Label-Efficient Sleep Stage Classification: A Comprehensive Evaluation [[Paper](https://arxiv.org/abs/2210.06286)][[Cite](#citation)]
### *by: Emadeldeen Eldele, Mohamed Ragab, Zhenghua Chen, Min Wu, Chee Keong Kwoh, and Xiaoli Li.*

## Abstract:
The past few years have witnessed a remarkable advance in deep learning for EEG-based sleep stage classification (SSC). 
However, the success of these models is attributed to possessing a massive amount of _labeled_ data for training, limiting their applicability in real-world scenarios. 
In such scenarios, sleep labs can generate a massive amount of data, but labeling these data can be expensive and time-consuming. 
Recently, the self-supervised learning (SSL) paradigm has shined as one of the most successful techniques to overcome the scarcity of labeled data.
In this paper, we evaluate the efficacy of SSL to boost the performance of existing SSC models in the few-labels regime.
We conduct a thorough study on three SSC datasets, and we find that fine-tuning the pretrained SSC models with only 5\% of labeled data can achieve competitive performance to the supervised training with full labels. Moreover, self-supervised pretraining helps SSC models to be more robust to data imbalance and domain shift problems.


## Datasets:
We used three public datasets in this study:
- Sleep-EDF
- SHHS
- ISRUC

We use the same preprocessing in [TS-TCC](https://github.com/emadeldeen24/TS-TCC/blob/main/data_preprocessing/sleep-edf/preprocess_sleep_edf.py).

To split the data into k-fold cross-validation and get the few label percentages, use `split_k-fold_and_few_labels.py` file.

To add a new dataset, include it in the "data" folder, add it to the `configs/hparam.py`
and `configs.data_configs.py` files. 
Also add a corresponding definition in the [trainer file](https://github.com/emadeldeen24/eval_ssl_ssc/blob/aba1d27fb0694146b3461b114874f5a5639cdc1b/trainer.py#L59).

## Self-supervised learning Algorithms
We used four SSL algorithms:
- ClsTran
- [SimCLR](http://proceedings.mlr.press/v119/chen20j/chen20j.pdf)
- [CPC](https://arxiv.org/abs/1807.03748)
- [TS-TCC](https://www.ijcai.org/proceedings/2021/0324.pdf)

To add a new SSL algorithm, include it in the `algorithms.py` file.


## Sleep stage classification models.
We used three SSC models:
- [DeepSleepNet](https://arxiv.org/abs/1703.04046)
- [AttnSleep](https://ieeexplore.ieee.org/document/9417097/)
- [1D-CNN](https://www.ijcai.org/proceedings/2021/0324.pdf)

The code of the adopted SSC models is included in `models/models.py`. Each SSC model is split into
two classes: feature extractor and temporal encoder. The classifier class is common among them all.

To add a new SSC model, say: "sleepX", include 2 classes:
"sleepX_fe" and "sleepX_temporal" in `models/models.py`.

## Training modes:
<ol>
    <li>Supervised: include "supervised" in the training mode.</li>
    <li>Self-supervised training: use "ssl"</li>
    <li>Fine-tuning with different label percentages: include "ft" in the training mode
name and use any other description for yourself (e.g., "ft_1per_labels")</li>
</ol>


## Results:
- The experiments of each dataset will be in a separate folder.
- Each run_description will be in a separate folder.
- For the K-fold settings, the overall results will be calculated after the last fold.
- Also, change ["4"](https://github.com/emadeldeen24/eval_ssl_ssc/blob/aba1d27fb0694146b3461b114874f5a5639cdc1b/trainer.py#L171) if you used different k-fold settings.


### Run the code:
either use `single_run.sh` to run a specific fold
or use `multiple_runs.sh` for multiple folds/multiple SSC models.


### Citation:
If you found this work useful for you, please consider citing it.
```
@article{emadeldeen2022eval,
  title={Self-supervised Learning for Label-Efficient Sleep Stage Classification: A Comprehensive Evaluation},
  author={Eldele, Emadeldeen and Ragab, Mohamed and Chen, Zhenghua and Wu, Min and Kwoh, Chee Keong and Li, Xiaoli},
  journal={arXiv preprint arXiv:2210.06286},
  year={2022}
}
```


## Contact
For any issues/questions regarding the paper or reproducing the results, please contact me.   
Emadeldeen Eldele   
School of Computer Science and Engineering (SCSE),   
Nanyang Technological University (NTU), Singapore.   
Email: emad0002{at}e.ntu.edu.sg   
