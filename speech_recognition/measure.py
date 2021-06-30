import tensorflow as tf


class SparseCategoricalCrossentropy(tf.keras.losses.Loss):
    """Normal sparse categorical crossentrophy with ignore index"""

    def __init__(
        self,
        ignore_index: int = 0,
        from_logits=True,
        reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        name="sparse_categorical_crossentropy",
    ):
        super().__init__(name=name, reduction=reduction)
        self.ignore_index = ignore_index
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=self.from_logits)
        loss = tf.boolean_mask(loss, y_true != self.ignore_index)
        return loss


class CTCLoss(tf.keras.losses.Loss):
    """Loss function to train DeepSpeech2"""

    def __init__(self, blank_index: int, pad_index: int = 0, name="ctc_loss"):
        super().__init__(name=name)
        self.blank_index = blank_index
        self.pad_index = pad_index

    def call(self, y_true, y_pred):
        batch_size = tf.shape(y_true)[0]
        sequence_length = y_pred.shape[1] or tf.shape(y_pred)[1]

        label_lengths = tf.math.count_nonzero(y_true != self.pad_index, axis=1)
        logit_lengths = tf.fill([batch_size], sequence_length)
        loss = tf.nn.ctc_loss(
            y_true, tf.cast(y_pred, tf.float32), label_lengths, logit_lengths, False, blank_index=self.blank_index
        )
        loss /= tf.cast(label_lengths, loss.dtype)
        return loss


class SparseCategoricalAccuracy(tf.keras.metrics.Metric):
    """Normal sparse categorical accuracy with ignore index"""

    def __init__(self, ignore_index: int = 0, name="accuracy"):
        super().__init__(name=name)

        self.ignore_index = ignore_index
        self.total_sum = self.add_weight(name="total_sum", initializer="zeros")
        self.total_count = self.add_weight(name="total_count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [tf.shape(y_true)[0], -1])
        accuracy = tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        accuracy = tf.boolean_mask(accuracy, y_true != self.ignore_index)
        if sample_weight is not None:
            accuracy = tf.multiply(accuracy, sample_weight)

        self.total_sum.assign_add(tf.reduce_sum(accuracy))
        self.total_count.assign_add(tf.cast(tf.shape(accuracy)[0], tf.float32))

        return accuracy

    def result(self):
        return self.total_sum / self.total_count
