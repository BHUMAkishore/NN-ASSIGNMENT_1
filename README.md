# NN-ASSIGNMENT_1
NAME:B.KISHORE BABU
ID:700752976
QUESTION_4.2
1.What patterns do you observe in the training and validation accuracy curves?
ANS)
Adam converges faster, with a steep rise in accuracy early on, but can overfit, leading to a plateau in validation accuracy.
SGD has a slower but steadier accuracy increase and better generalization with smaller training-validation accuracy gaps.
Adam reduces loss faster due to adaptive learning rates, while SGD has a higher initial loss and more fluctuations in accuracy.
A large gap between training and validation accuracy indicates overfitting, more noticeable in Adam.

2.How can you use TensorBoard to detect overfitting?
ans)
Accuracy Curves: Monitor training and validation accuracy. If training accuracy rises while validation accuracy plateaus or drops, it suggests overfitting.
Loss Curves: Track both training and validation loss. Overfitting occurs when training loss decreases but validation loss increases.
Gradients & Weights: Check for exploding or vanishing gradients that may affect model generalization.
Setup:
Log metrics using tf.summary.
Run tensorboard --logdir=logs/ to visualize.
Watch for diverging curves to detect overfitting.

3.What happens when you increase the number of epochs?
ans)
Increasing epochs can:
Improve Training: More time for the model to learn.
Risk Overfitting: The model may memorize data, harming generalization.
Better Generalization (with early stopping): Allows more learning without overfitting.
Increase Training Time: More epochs mean longer training.
