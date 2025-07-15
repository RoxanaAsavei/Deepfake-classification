# Deepfake Image-Classification Coursework

*For a complete walk-through of the data analysis, hyper-parameter sweeps, learning curves, and confusion matrices please refer to the [report](https://github.com/RoxanaAsavei/Deepfake-classification/blob/main/Asavei_Roxana_251_doc.pdf) in the repository root.*

---

## Project purpose

The code and models here were developed for the 2nd-year **Artificial Intelligence** machine-learning competition, whose task was to recognise which of **five generative models** produced a given synthetic image of animals performing unusual activities – a 5-class deepfake classification problem.

---
## Implemented models

### K-Nearest Neighbours (KNN)  
* **Feature extraction** – each image is represented by a 512-bin RGB colour histogram (8 × 8 × 8), L1-normalised and flattened. 
* **Best hyper-parameters** – a grid search found the optimal setting at **k = 7** neighbours using Manhattan (L1) distance.
* **Performance insights** – the confusion matrix highlights class-specific weaknesses: class 4 is recognised correctly only 50 % of the time, and ~30 % of its samples are mislabelled as class 1. 
* **Results** – achieves **68.16 % validation accuracy**. 
---

### Convolutional Neural Network (CNN)  
* **Architecture** – three Conv → MaxPool → BatchNorm blocks followed by a Conv → AveragePool → BatchNorm block, then GlobalAveragePooling and a dense softmax head. Implemented in both Keras and PyTorch with no significant accuracy gap. 
* **Training recipe** – images resized to 224 × 224 and scaled /255; Adam optimiser with learning-rate = 0.01 and momentum = 0.9. After 30 epochs without augmentation, training continues with random 15° rotations, ±10 % shifts & zoom, and horizontal flips; batch-size = 32 proves best.   
* **Results** – achieves **90.24 % validation accuracy** and **91.52 % on the hidden test set**.
* **Error pattern** – most residual mistakes involve confusing classes 1 and 4, mirroring the KNN behaviour. 

