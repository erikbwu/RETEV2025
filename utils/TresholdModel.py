class Trashy:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def predict(self, x):
        """
        Predicts the class based on a threshold.

        Parameters:
            x (numpy.ndarray): The input data.

        Returns:
            numpy.ndarray: The predicted class (0 or 1).
        """
        return 0