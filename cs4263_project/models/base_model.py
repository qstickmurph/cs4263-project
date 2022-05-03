__all__ = [
    "BaseModel",
]

import tensorflow as tf

from ..data.preprocessing import restore_google_trends

class BaseModel(tf.keras.Sequential):
    def adjusted_evaluate(self, x, y):
        """
        Parameters:
            x: flat tensor of features
            y: flat tensor of expected labels
        """
        prediction = self(x)
        mae = tf.reduce_mean(tf.abs(y- prediction))
        rmse = tf.sqrt(tf.reduce_mean((y- prediction) ** 2))
        
    def get_predictions_df(self):
        pass

    def save_model(self):
        self.__super__()