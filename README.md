# DeepAFP: an effective computational framework for identifying antifungal peptides based on deep learning



Fungal infections have become a significant global health issue, affecting millions worldwide. Antifungal peptides (AFPs) have emerged as a promising alternative to conventional antifungal drugs due to their low toxicity and low propensity for inducing resistance. In this study, we developed a deep learning-based framework called **DeepAFP** to efficiently identify AFPs. DeepAFP fully leverages and mines composition information, evolutionary information, and physicochemical properties of peptides by employing combined kernels from multiple branches of convolutional neural network (CNN) with bi-directional long short-term memory (Bi-LSTM) layers. In addition, DeepAFP integrates a transfer learning strategy to obtain efficient representations of peptides for improving model performance. DeepAFP demonstrates strong predictive ability on carefully curated datasets, achieving an accuracy of 93.29% and an F1-score of 93.45% on the DeepAFP-Main dataset. The experimental results show that DeepAFP outperforms existing AFP prediction tools, achieving state-of-the-art performance. Finally, we provide a downloadable AFP prediction tool to meet the demands of large-scale prediction and facilitate the use of our framework by researchers unfamiliar with deep learning. Our proposed framework can accurately identify AFPs in a short time without requiring significant human and material resources. Therefore, we believe that DeepAFP can accelerate the development of AFPs and make a contribution to the treatment of fungal infections. Furthermore, our method can provide new perspectives for other biological sequence analysis tasks.


## Datasets for this study

We provided our datasets (**DeepAFP-main, DeepAFP-Set 1 and DeepAFP-Set 1**) and you can find them [here](https://github.com/lantianyao/DeepAFP/tree/main/dataset "here").


## Feature extraction of peptides

We provide codes for peptide feature extraction, including **amino acid features (Binary profile, BLOSUM62 and Z-scale)** and **BERT encodings**.

For **amino acid features (including Binary profile, BLOSUM62 and Z-scale)** extraction, you can find an example [here](https://github.com/lantianyao/DeepAFP/blob/main/Extract%20peptide%20features.ipynb "here").

As for the extraction of **BERT encodings**, you can find an example [here](https://github.com/lantianyao/DeepAFP/blob/main/feature_extraction_tape.py "here").

The extracted amino acid features (including Binary profile, BLOSUM62 and Z-scale) and BERT encodings will be used as input to the deep learning models.

## Use DeepAFP
- If you would like to use our trained models, please refer to: [Use trained models.ipynb.](https://github.com/lantianyao/DeepAFP/blob/main/Use%20trained%20models.ipynb "Use trained models.ipynb.")
- If you want to train the model with your own dataset, please refer to: [Training](https://github.com/lantianyao/DeepAFP/tree/main/Training "Training")，which is based on Keras and Tensorflow, please make sure you have them installed before using our codes.
- In addition , we provide a downloadable prediction tool for Windows systems, if you want to use it, please refer to [here](https://awi.cuhk.edu.cn/dbAMP/DeepAFP.html "here").


**If you have any questions, please feel free to contact me.**
Name: Lantian Yao
Email: lantianyao@link.cuhk.edu.cn



