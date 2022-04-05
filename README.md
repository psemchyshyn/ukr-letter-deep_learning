# ukr-letter-deep_learning

## What was done

### Basic - classification part

The results in terms of metrics are the following (after training the model for 100 epochs (till convergencce))

{'accuracy': 0.8062674715635242, 'f1': 0.8042842430743762, 'recall': 0.8062674715635242, 'precision': 0.8493257370827263}

You can look at **losses_classification.png** to see train/valiation loss (binary cross-entropy) and accuracy changes across the epochs

The results are quite good for a very basic convolutional network (almost identical to MNIST classification network architecture)

*Note: no data augmentation, upsampling or downsampling techniques were used*

### Custom - anomaly detection

The idea is to utilize the ideas of variational autoencoders for an unsupervised anomaly detection task. I choose the letter "Ї" (the sparsest class ~300 samples) to be considered as anomalous.
The training is performed on normal samples only, so that the autoencoder learns how to reconstruct these normal samples. With this assumption, the model won't
be able to reconstruct anomalous letters while testing (because it hasn't seen them during training), producing high reconstruction error for them.

The model produced very unsatisfying accuracy of ~55%, which is slightly better than random guess. The explanation is that the architecture for both encoder and decoder 
was chosen to be quite trivial, therefore the model wasn't actually very good at reconstructing even some normal samples, while it was reconstructing
anomalous "Ї" samples as "Т", which indeed are quite similiar(reconstruction error wasn't quite high for them).

You can see the reconstructed samples for normal and abnormal data im **/images** folder.
Thes losses can be seen in **losses_anomaly.png**


## Repository organization

* **/models** - models architecture
* **anomaly_detection.py** - training and testing of custom part (unsupervised anomaly detection)
* **classification.py** - training and testing of basic part (classification)
* **data.py** - creating custom dataset classes and the preprocessing step
* **clean.py** - adapts original dataset for training (leaves only capital letters)
* **/checkpoints** - saved model state at every 10 epoch pass (classification)
* **/checkpoints_anomalies** - saved model state at every 10 epoch pass (unsupervised anomaly detection)

## Important notes

* running **anomaly_detection.py** or **classification.py** will firstly try to upload trained (or partially trained) model from corresponding checkpoints directory. If there aren't any saved states, the training will start from scratch
* training (in both parts) is performed only on a subset of original dataset (capital letters, i.e A_1). You can run **clean.py** to adapt original dataset for training


