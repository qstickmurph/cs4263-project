__all__ = [
    "get_predictions_df"
]

import numpy as np
import pandas as pd

from .preprocessing import restore_nymex

def get_predictions_df(model, dataset, label_width, labels, index):
    predictions = np.ndarray((len(index), label_width, len(labels)))
    dataset = dataset.batch(1)

    for i, tensor in enumerate(iter(dataset)):
        predictions[i,:,:] = model(tensor[0]).numpy()

    # get rid of time dim
    if label_width == 1:
        predictions = predictions.reshape((len(index), len(labels)))
    else: # compress time dim
        pass

    predictions_df = pd.DataFrame(predictions, index=index, columns=labels)
    predictions_df = restore_nymex(predictions_df)
    return predictions_df