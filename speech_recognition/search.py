import tensorflow as tf

from .model import LAS


class Searcher:
    """Provide search functions for LAS model"""

    def __init__(self, model: LAS, max_token_length: int, bos_id: int, eos_id: int, pad_id: int = 0):
        """
        :param model: LAS model instance.
        :param max_token_length: max sequence length of decoded sequences.
        :param bos_id: bos id for decoding.
        :param eos_id: eos id for decoding.
        :param pad_id: when a sequence is shorter thans other sentences, the back token ids of the sequence is filled pad id.
        """
        self.model = model
        self.max_token_length = max_token_length
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id

    @tf.function
    def greedy_search(self, audio_input: tf.Tensor) -> tf.Tensor:
        """
        Generate sentences using decoder by greedy searching.

        :param audio_input: model. model inputs [BatchSize, TimeStep, DimAudio, 3].
        :return: generated tensor shaped. and ppl value of each generated sentences
        """
        batch_size = tf.shape(audio_input)[0]
        decoder_input = tf.fill([batch_size, 1], self.bos_id)
        log_perplexity = tf.fill([batch_size, 1], 0.0)
        sequence_lengths = tf.fill([batch_size, 1], self.max_token_length)
        is_ended = tf.zeros([batch_size, 1], tf.bool)

        def _cond(decoder_input, is_ended, log_perplexity, sequence_lengths, states):
            return tf.shape(decoder_input)[1] < self.max_token_length and not tf.reduce_all(is_ended)

        def _body(decoder_input, is_ended, log_perplexity, sequence_lengths, states):
            # [BatchSize, VocabSize]
            output, *states = self.model.attend_and_speller(audio_output, decoder_input[:, -1], mask, states)
            output = tf.nn.log_softmax(output, axis=1)

            # [BatchSize, 1]
            log_probs, new_tokens = tf.math.top_k(output)
            log_probs, new_tokens = tf.cast(log_probs, log_perplexity.dtype), tf.cast(new_tokens, tf.int32)
            log_perplexity = tf.where(is_ended, log_perplexity, log_perplexity + log_probs)
            new_tokens = tf.where(is_ended, self.pad_id, new_tokens)
            is_ended = tf.logical_or(is_ended, new_tokens == self.eos_id)
            sequence_lengths = tf.where(new_tokens == self.eos_id, tf.shape(decoder_input)[1] + 1, sequence_lengths)

            # [BatchSize, DecoderSequenceLength + 1]
            decoder_input = tf.concat((decoder_input, new_tokens), axis=1)

            return decoder_input, is_ended, log_perplexity, sequence_lengths, states

        # Encoding
        audio_output, mask, *states = self.model.listener(audio_input)

        # Decoding
        decoder_input, is_ended, log_perplexity, sequence_lengths, states = tf.while_loop(
            _cond,
            _body,
            [decoder_input, is_ended, log_perplexity, sequence_lengths, states],
            shape_invariants=[
                tf.TensorSpec([None, None], tf.int32),
                tf.TensorSpec(is_ended.get_shape(), is_ended.dtype),
                tf.TensorSpec(log_perplexity.get_shape(), log_perplexity.dtype),
                tf.TensorSpec(sequence_lengths.get_shape(), sequence_lengths.dtype),
                [
                    tf.TensorSpec([None, None]),
                    tf.TensorSpec([None, None]),
                ],
            ],
        )

        perplexity = tf.squeeze(
            tf.pow(tf.exp(log_perplexity), tf.cast(-1 / sequence_lengths, log_perplexity.dtype)), axis=1
        )
        return decoder_input, perplexity

    @tf.function
    def beam_search(
        self,
        audio_input: tf.Tensor,
        beam_size: int,
        alpha: float = 1,
        beta: int = 32,
    ) -> tf.Tensor:
        """
        Generate sentences using decoder by beam searching.

        :param audio_input: model. model inputs [BatchSize, TimeStep, DimAudio, 3].
        :param beam_size: beam size for beam search.
        :param alpha: length penalty control variable
        :param beta: length penalty control variable, meaning minimum length.
        :return: generated tensor shaped. and ppl value of each generated sentences
            decoder_input: (BatchSize, BeamSize, SequenceLength)
            perplexity: (BatchSize, BeamSize)
        """
        batch_size = tf.shape(audio_input)[0]
        decoder_input = tf.fill([batch_size, 1], self.bos_id)
        log_perplexity = tf.fill([batch_size, 1], 0.0)

        def _to_sequence_lengths(decoder_single_input):
            eos_indices = tf.where(decoder_single_input == self.eos_id)
            if tf.size(eos_indices) == 0:
                return tf.size(decoder_single_input, tf.int32)
            return tf.cast(tf.math.reduce_min(eos_indices) + 1, tf.int32)

        def get_sequnce_lengths(decoder_input):
            original_shape = tf.shape(decoder_input)
            decoder_input = tf.reshape(decoder_input, (-1, original_shape[-1]))
            sequence_lengths = tf.map_fn(_to_sequence_lengths, decoder_input)
            return tf.reshape(sequence_lengths, original_shape[:-1])

        def has_eos(decoder_input):
            return tf.reduce_any(decoder_input == self.eos_id, axis=-1)

        def _cond(audio_output, decoder_input, mask, log_perplexity, states):
            return tf.shape(decoder_input)[1] < self.max_token_length and tf.reduce_any(
                tf.logical_not(has_eos(decoder_input))
            )

        def _body(audio_output, decoder_input, mask, log_perplexity, states):
            # [BatchSize, VocabSize]
            output, *states = self.model.attend_and_speller(audio_output, decoder_input[:, -1], mask, states)
            output = tf.nn.log_softmax(output, axis=1)

            # [BatchSize, BeamSize] at first, [BatchSize * BeamSize, BeamSize] after second loops
            log_probs, new_tokens = tf.math.top_k(output, k=beam_size)

            # log_probs: [BatchSize, BeamSize] at first, [BatchSize, BeamSize ** 2] after second loops
            # new_tokens: [BatchSize, 1] at first, [BatchSize * BeamSize, 1] after second loops
            log_probs, new_tokens = tf.reshape(log_probs, [batch_size, -1]), tf.reshape(new_tokens, [-1, 1])
            is_end_sequences = tf.reshape(tf.repeat(has_eos(decoder_input), beam_size, axis=0), [batch_size, -1])
            log_probs = tf.where(is_end_sequences, tf.cast(0.0, log_probs.dtype), log_probs)
            log_probs += tf.cast(tf.repeat(log_perplexity, beam_size, axis=1), log_probs.dtype)

            # Generate first token
            if tf.shape(decoder_input)[1] == 1:
                audio_output = tf.repeat(audio_output, beam_size, axis=0)
                mask = tf.repeat(mask, beam_size, axis=0)
                new_states = []
                for state in states:
                    new_states += [tf.repeat(state, beam_size, axis=0)]
                states = new_states

                # [BatchSize * BeamSize, 2]
                decoder_input = tf.concat([tf.fill([batch_size * beam_size, 1], self.bos_id), new_tokens], axis=1)
                log_perplexity = tf.cast(log_probs, log_perplexity.dtype)
                return audio_output, decoder_input, mask, log_perplexity, states
            else:
                # [BatchSize * BeamSize, BeamSize, DecoderSequenceLength + 1]
                decoder_input = tf.reshape(
                    tf.concat((tf.repeat(decoder_input, beam_size, axis=0), new_tokens), axis=1),
                    [batch_size, beam_size * beam_size, -1],
                )

            length_penalty = tf.pow((1 + get_sequnce_lengths(decoder_input)) / (1 + beta), alpha)
            length_penalty = tf.cast(tf.reshape(length_penalty, tf.shape(log_probs)), log_probs.dtype)
            # [BatchSize, BeamSize]
            _, top_indices = tf.math.top_k(log_probs * length_penalty, k=beam_size)

            # [BatchSize * BeamSize, 2]
            indices_for_decoder_input = tf.concat(
                [
                    tf.reshape(tf.repeat(tf.range(batch_size), beam_size), [batch_size * beam_size, 1]),
                    tf.reshape(top_indices, [batch_size * beam_size, 1]),
                ],
                axis=1,
            )

            # [BatchSize * BeamSize, DecoderSequenceLength]
            decoder_input = tf.gather_nd(decoder_input, indices_for_decoder_input)
            log_perplexity = tf.cast(tf.gather_nd(log_probs, indices_for_decoder_input), log_perplexity.dtype)
            log_perplexity = tf.reshape(log_perplexity, [batch_size, beam_size])

            return audio_output, decoder_input, mask, log_perplexity, states

        # Encoding
        audio_output, mask, *states = self.model.listener(audio_input)

        # Decoding
        audio_output, decoder_input, mask, log_perplexity, states = tf.while_loop(
            _cond,
            _body,
            [audio_output, decoder_input, mask, log_perplexity, states],
            shape_invariants=[
                tf.TensorSpec([None, None, None], tf.float32),
                tf.TensorSpec([None, None], tf.int32),
                tf.TensorSpec([None, None], tf.bool),
                tf.TensorSpec([None, None]),
                [
                    tf.TensorSpec([None, None]),
                    tf.TensorSpec([None, None]),
                ],
            ],
        )

        decoder_input = tf.reshape(decoder_input, [batch_size, beam_size, -1])
        sequence_lengths = get_sequnce_lengths(decoder_input)
        decoder_input = tf.where(
            tf.sequence_mask(sequence_lengths, tf.shape(decoder_input)[2]), decoder_input, self.pad_id
        )
        perplexity = tf.pow(tf.exp(log_perplexity), tf.cast(-1 / sequence_lengths, log_perplexity.dtype))

        return decoder_input, perplexity
