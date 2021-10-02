import os

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np
import io
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix, plot_roc_curve, roc_curve, auc, RocCurveDisplay

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

class PerformanceVisualizationCallback(Callback):
    def __init__(self, model, validation_data, image_dir):
        super().__init__()
        self.model = model
        self.validation_data = validation_data
        
        os.makedirs(image_dir, exist_ok=True)
        self.image_dir = image_dir

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.validation_data, steps=len(self.validation_data))
        y_pred_class = np.argmax(y_pred, axis=1)
        y_true = np.argmax(self.validation_data.targets[self.validation_data.start_index:], axis=1)
        
        cm = confusion_matrix(y_true, y_pred_class, normalize='pred') # normalize='pred', 'true', 'all'
        #         plot_confusion_matrix(y_true, y_pred_class, ax=ax)
        #  ‘1’ is for up, ‘2’ is for stationary and ‘3’ is for down.
        cmd = ConfusionMatrixDisplay(cm, display_labels=['Up','Stationary', 'Down'])
        # plot and save confusion matrix
        fig, ax = plt.subplots(figsize=(16,12))
        cmd.plot(ax=ax, cmap=plt.cm.Blues, values_format='g')
#         fig.savefig(os.path.join(self.image_dir, f'confusion_matrix_epoch_{epoch}'))
        
        # Log the confusion matrix as an image summary
        file_writer_cm = tf.summary.create_file_writer(self.image_dir)
        with file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", plot_to_image(fig) , step=epoch)

        # plot and save roc curve
#         fpr, tpr, thresholds = roc_curve(y_true, y_pred_class)
#         roc_auc = auc(fpr, tpr)
#         display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Price movement predictor')
# #         plot_roc_curve(y_true, y_pred, ax=ax)
#         fig, ax = plt.subplots(figsize=(16,12))
#         display.plot(ax=ax)
#         fig.figure_.savefig(os.path.join(self.image_dir, f'roc_curve_epoch_{epoch}'))
