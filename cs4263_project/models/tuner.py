__all__ = [
    "BayestianOptimizationWithValidation",
]
import warnings

import tensorflow as tf
import keras_tuner as kt
from keras_tuner.engine import trial as trial_module
from keras_tuner.engine import tuner_utils

class BayestianOptimizationWithValidation(kt.BayesianOptimization):
    # Just use parent's constructor
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Rewrite search and add the ability for a validation set
    def run_trial(self, trial, *args, **kwargs):
        """Evaluates a set of hyperparameter values.
        This method is called multiple times during `search` to build and
        evaluate the models with different hyperparameters and return the
        objective value.
        Example:
        You can use it with `self.hypermodel` to build and fit the model.
        ```python
        def run_trial(self, trial, *args, **kwargs):
            hp = trial.hyperparameters
            model = self.hypermodel.build(hp)
            return self.hypermodel.fit(hp, model, *args, **kwargs)
        ```
        You can also use it as a black-box optimizer for anything.
        ```python
        def run_trial(self, trial, *args, **kwargs):
            hp = trial.hyperparameters
            x = hp.Float("x", -2.0, 2.0)
            y = x * x + 2 * x + 1
            return y
        ```
        Args:
            trial: A `Trial` instance that contains the information needed to
                run this trial. Hyperparameters can be accessed via
                `trial.hyperparameters`.
            *args: Positional arguments passed by `search`.
            **kwargs: Keyword arguments passed by `search`.
        Returns:
            A `History` object, which is the return value of `model.fit()`, a
            dictionary, a float, or a list of one of these types.
            If return a dictionary, it should be a dictionary of the metrics to
            track. The keys are the metric names, which contains the
            `objective` name. The values should be the metric values.
            If return a float, it should be the `objective` value.
            If evaluating the model for multiple times, you may return a list
            of results of any of the types above. The final objective value is
            the average of the results in the list.
        """
        # Not using `ModelCheckpoint` to support MultiObjective.
        # It can only track one of the metrics to save the best model.
        model_checkpoint = tuner_utils.SaveBestEpoch(
            objective=self.oracle.objective,
            filepath=self._get_checkpoint_fname(trial.trial_id),
        )
        original_callbacks = kwargs.pop("callbacks", [])

        # Run the training process multiple times.
        histories = []
        for execution in range(self.executions_per_trial):
            copied_kwargs = copy.copy(kwargs)
            callbacks = self._deepcopy_callbacks(original_callbacks)
            self._configure_tensorboard_dir(callbacks, trial, execution)
            callbacks.append(tuner_utils.TunerCallback(self, trial))
            # Only checkpoint the best epoch across all executions.
            callbacks.append(model_checkpoint)
            copied_kwargs["callbacks"] = callbacks
            obj_value = self._build_and_fit_model(trial, *args, **copied_kwargs)

            histories.append(obj_value)
        return histories