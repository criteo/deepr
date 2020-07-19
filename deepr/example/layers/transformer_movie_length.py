# pylint: disable=no-value-for-parameter,invalid-name,unexpected-keyword-arg
"""Transformer Model"""

import logging
import os
import random
from typing import Tuple, List, Any, NamedTuple, Optional

import pandas as pd
import numpy as np
import tensorflow as tf

import deepr as dpr


LOGGER = logging.getLogger(__name__)


class Timeline(NamedTuple):
    uid: str
    movie_ids: List[int]

    def split(self, target_ratio: float = 0.2):
        start_target_index = int(len(self.movie_ids) * (1 - target_ratio))
        input_part = Timeline(
            uid=self.uid,
            movie_ids=self.movie_ids[:start_target_index],
        )
        target_part = Timeline(
            uid=self.uid,
            movie_ids=self.movie_ids[start_target_index:],
        )
        return input_part, target_part


class Record(NamedTuple):
    uid: str
    inputPositives: List[int]
    targetPositives: List[int]
    targetNegatives: List[int]


def upload_ml20m_dataset(path_ratings: str, min_rating: float, min_length: int) -> List[Timeline]:
    """
    load ml20m dataset and apply filters:
        keep movies with ratings > `min_rating`
        keep users with number of movies > `min_length`
    """

    def _sort_list_by_other_list(main_list: List[Any], key_list: List[Any]):
        return [item for key, item in sorted(zip(key_list, main_list))]

    ratings_data = pd.read_csv(path_ratings, sep=",")
    ratings_data = ratings_data[ratings_data.rating >= min_rating]
    grouped_data = ratings_data.groupby("userId").agg(list).reset_index()
    grouped_data = grouped_data[grouped_data.rating.map(len) >= min_length]
    timelines: List[Timeline] = []
    for index, row in grouped_data.iterrows():
        timeline = Timeline(
            uid=str(row.userId),
            movie_ids=_sort_list_by_other_list(row.movieId, key_list=row.timestamp)
        )
        timelines.append(timeline)
    return timelines


def split_dataset(timelines: List[Any], validation_ratio: float = 0.2, seed: int = 2020):
    random.seed(seed)
    random.shuffle(timelines)
    delimiter = int(len(timelines) * validation_ratio)
    train = timelines[delimiter:]
    validation = timelines[:delimiter]
    return train, validation


def negative_samples(num_examples: int, available_items, nb_negatives: int = 8):
    """ generate negative samples by group uniformly"""
    sample = []
    for _ in range(num_examples):
        random_items = random.sample(available_items, nb_negatives)
        sample.append(random_items)
    return sample


def generate_negative_sampling(
        timelines: List[Timeline],
        target_ratio,
        nb_negatives,
) -> List[Record]:
    """
    convert list of `Timeline` to list of `Record` and generation of negative samples
    """

    # get all movies that will be used for negative sampling
    available_movies = set()
    for timeline in timelines:
        for movie_id in timeline.movie_ids:
            available_movies.add(movie_id)

    # split each timeline into input and target parts and generate negative samples for each movie
    records = []
    for timeline in timelines:
        input_timeline, target_timeline = timeline.split(target_ratio=target_ratio)
        record = Record(
            uid=input_timeline.uid,
            inputPositives=input_timeline.movie_ids,
            targetPositives=target_timeline.movie_ids,
            targetNegatives=negative_samples(
                len(target_timeline.movie_ids), available_movies, nb_negatives),
        )
        records.append(record)
    return records


def prepare_dataset(
        data_dir: str,
        min_rating: int = 4,
        min_length: int = 5,
        validation_ratio: float = 0.2,
        nb_negatives: int = 8,
        target_ratio: float = 0.2
):
    path_ratings = os.path.join(data_dir, "ratings.csv")

    # upload the dataset
    timelines: List[Timeline] = upload_ml20m_dataset(
        path_ratings=path_ratings, min_rating=min_rating, min_length=min_length
    )

    # split into train and validation
    train, validation = split_dataset(timelines, validation_ratio=validation_ratio, seed=2020)

    # split timelines on input/target and generate negative examples
    train_dataset: List[Record] = generate_negative_sampling(train, target_ratio, nb_negatives)
    validation_dataset: List[Record] = generate_negative_sampling(validation, target_ratio, nb_negatives)
    return train_dataset, validation_dataset


def get_input_fn(dataset):
    def train_generator():
        for item in dataset:
            yield item._asdict()

    return dpr.readers.GeneratorReader(
        train_generator,
        output_types={
            "uid": tf.string,
            "inputPositives": tf.int64,
            "targetPositives": tf.int64,
            "targetNegatives": tf.int64,
        },
        output_shapes={
            "uid": [],
            "inputPositives": tf.TensorShape([None]),
            "targetPositives": tf.TensorShape([None]),
            "targetNegatives": tf.TensorShape([None, None]),
        },
    )


def get_prepro_fn(batch_size: int = 16, buffer_size: int = 10, epochs: Optional[int] = None, max_input_size: int = 10000, max_target_size: int = 1000):

    fields = [
        dpr.Field(name="uid", shape=(None,), dtype=tf.string),
        dpr.Field(name="inputPositives", shape=(None,), dtype=tf.int64),
        dpr.Field(name="targetPositives", shape=(None,), dtype=tf.int64),
        dpr.Field(name="targetNegatives", shape=(None, None), dtype=tf.int64),
    ]
    return dpr.prepros.Serial(
        (dpr.prepros.Map(dpr.layers.SliceLast(max_input_size, inputs="inputPositives", outputs=key)) for key in ["inputPositives"]),
        (dpr.prepros.Map(dpr.layers.SliceFirst(max_target_size, inputs=key, outputs=key)) for key in ["targetPositives", "targetNegatives"]),
        (dpr.prepros.PaddedBatch(batch_size=batch_size, fields=fields)),
        dpr.prepros.Repeat(epochs, modes=[dpr.TRAIN]),
        dpr.prepros.Prefetch(buffer_size),
    )


def TransformerModel(
    vocab_size: int,
    train_embeddings: bool,
    dim: int,
    heads: int = 4,
    encoding_blocks: int = 2,
    d_model: int = 512,
    residual_connection: bool = True,
    use_layer_normalization: bool = True,
    event_dropout_rate: float = 0.0,
    ff_dropout_rate: float = 0.0,
    use_feedforward: bool = True,
    ff_normalization: bool = False,
    scale: bool = False,
    use_positional_encoding: bool = True,
    learnable_positional_encoding: bool = True,
    use_subsequent_mask: bool = True,
    inputs: str = None,
    outputs: str = None,
) -> dpr.layers.Layer:
    """ Build Transformer Model"""

    _inputs = ("inputPositives", "targetPositives", "targetNegatives")
    _outputs = ("userEmbeddings", "targetPositivesEmbeddings", "targetNegativesEmbeddings")

    return dpr.layers.Sequential(
        dpr.layers.Select(n_in=3, inputs=inputs if inputs else _inputs, outputs=_inputs),
        # Embed input and target products (model specific)
        dpr.layers.Embedding(
            variable_name="product_embeddings",
            shape=(vocab_size, dim),
            num_shards=5,
            trainable=train_embeddings,
            inputs="inputPositives",
            outputs="inputPositivesEmbeddings",
        ),
        dpr.layers.Embedding(
            variable_name="product_embeddings",
            shape=(vocab_size, dim),
            num_shards=5,
            trainable=train_embeddings,
            inputs="targetPositives",
            outputs="targetPositivesEmbeddings",
            reuse=True,
        ),
        dpr.layers.Embedding(
            variable_name="product_embeddings",
            shape=(vocab_size, dim),
            num_shards=5,
            trainable=train_embeddings,
            inputs="targetNegatives",
            outputs="targetNegativesEmbeddings",
            reuse=True,
        ),
        # Compute user embeddings
        dpr.layers.SpatialDropout1D(
            inputs="inputPositivesEmbeddings",
            outputs="inputPositivesEmbeddingsDropout",
            dropout_rate=event_dropout_rate,
        ),
        CreateMask(inputs="inputPositives", outputs="mask", use_look_ahead_mask=use_subsequent_mask),
        ScaleLayer(
            inputs="inputPositivesEmbeddingsDropout", outputs="inputEnc", multiplier=d_model ** 0.5, scale=scale
        ),
        PositionalEncoding(
            inputs="inputEnc",
            outputs="inputEnc",
            use_positional_encoding=use_positional_encoding,
            learnable=learnable_positional_encoding,
        ),
        (
            dpr.layers.Sequential(
                MultiheadAttentionLayer(
                    inputs=("inputEnc", "mask"),
                    outputs="inputEnc",
                    heads=heads,
                    d_model=d_model,
                    residual_connection=residual_connection,
                    emb_dim=dim,
                    use_layer_normalization=use_layer_normalization,
                    block_id=block_id,
                ),
                FeedForwardLayer(
                    inputs="inputEnc",
                    outputs="inputEnc",
                    inner_units=d_model,
                    readout_units=d_model,
                    ff_dropout_rate=ff_dropout_rate,
                    emb_dim=dim,
                    use_feedforward=use_feedforward,
                    ff_normalization=ff_normalization,
                    block_id=block_id,
                ),
            )
            for block_id in range(encoding_blocks)
        ),
        dpr.layers.SliceLastPadded(
            inputs=("inputEnc", "inputPositives"), outputs="userEmbeddings", padded_value=-1
        ),
        dpr.layers.Select(inputs=_outputs, outputs=outputs if outputs else _outputs),
    )


@dpr.layers.layer(n_in=1, n_out=1)
def FeedForwardLayer(
    tensors: tf.Tensor,
    mode: str,
    inner_units: int,
    readout_units: int,
    emb_dim: int,
    use_feedforward: bool,
    ff_normalization: bool,
    ff_dropout_rate: float,
    block_id: int,
):
    """ FeedForward Layer """
    if not use_feedforward:
        return tensors

    with tf.variable_scope(f"feedforward_{block_id}"):
        outputs = dpr.layers.Sequential(
            dpr.layers.Dropout(dropout_rate=ff_dropout_rate),
            dpr.layers.Conv1d(filters=inner_units, kernel_size=1, activation=tf.nn.relu, use_bias=True),
            dpr.layers.Dropout(dropout_rate=ff_dropout_rate),
            dpr.layers.Conv1d(filters=readout_units, kernel_size=1, activation=None, use_bias=True),
            dpr.layers.Dropout(dropout_rate=ff_dropout_rate),
            dpr.layers.Dense(units=emb_dim),
        )(tensors, mode)

        outputs += tensors  # residual connection

        if ff_normalization:
            outputs = NormalizationLayer()(outputs, mode)

        return outputs


@dpr.layers.layer(n_in=1, n_out=1)
def NormalizationLayer(tensors: tf.Tensor, mode: str, epsilon=1e-8):
    """ Normalization Layer """
    # pylint: disable=unused-argument
    with tf.variable_scope("layer_normalization"):
        params_shape = tensors.get_shape()[-1:]

        mean, variance = tf.nn.moments(tensors, [-1], keep_dims=True)

        beta = tf.get_variable("beta", shape=params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", shape=params_shape, initializer=tf.ones_initializer())

        normalized = (tensors - mean) / ((variance + epsilon) ** 0.5)
        outputs = gamma * normalized + beta

    return outputs


@dpr.layers.layer(n_in=1, n_out=1)
def ScaleLayer(tensors: tf.Tensor, mode: str, multiplier: float, scale: bool):
    # pylint: disable=unused-argument
    if scale:
        return tf.multiply(tensors, multiplier)
    else:
        return tensors


@dpr.layers.layer(n_in=1, n_out=1)
def PositionalEncoding(
    tensors: tf.Tensor, mode: str, use_positional_encoding: bool, max_sequence_length=10000, learnable=False
):
    """ Positional Encoding Layer

        Parameters
        ----------
        tensors : tf.Tensor, 2 dimensions
            Input tensor
        use_positional_encoding : bool
            Use this layer in case of True, skipp in case of False
        max_sequence_length: int
            Expected that input tensor length doesn't exceed the `max_sequence_length` limit
        learnable: bool
            Train / not train position encoding

        Returns
        -------
        out : tf.Tensor, 2 dimensions
            Tensor with applied positional encoding
    """
    # pylint: disable=unused-argument
    if not use_positional_encoding:
        return tensors

    with tf.variable_scope("positional_encoding", reuse=tf.AUTO_REUSE):  # AUTO_REUSE is important here

        emb_dim = tensors.get_shape().as_list()[-1]
        bs, sl = tf.shape(tensors)[0], tf.shape(tensors)[1]

        # position indices
        position_ind = tf.tile(tf.expand_dims(tf.range(sl), 0), [bs, 1])  # (bs, sl)

        if learnable:
            lookup_table = tf.get_variable(
                "lookup_table",
                dtype=tf.float32,
                shape=[max_sequence_length, emb_dim],
                regularizer=tf.contrib.layers.l2_regularizer(0.0),
            )
            outputs = tf.nn.embedding_lookup(lookup_table, position_ind, partition_strategy="div")
        else:
            position_enc = np.array(
                [
                    [pos / np.power(10000, (i - i % 2) / emb_dim) for i in range(emb_dim)]
                    for pos in range(max_sequence_length)
                ]
            )

            position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
            position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
            position_enc = tf.convert_to_tensor(position_enc, tf.float32)  # (max_sequence_length, emb_dim)

            outputs = tf.nn.embedding_lookup(position_enc, position_ind, partition_strategy="div")

        return tensors + tf.to_float(outputs)


@dpr.layers.layer(n_in=1, n_out=1)
def CreateMask(tensors: tf.Tensor, mode: str, use_look_ahead_mask: bool):
    """ Create padding and look ahead masks, and combine them"""
    padding_mask = dpr.layers.PaddingMask(padded_value=-1)(tensors, mode)
    padding_mask = padding_mask[:, tf.newaxis, tf.newaxis, :]  # (bs, 1, sl, sl)

    if use_look_ahead_mask:
        look_ahead_mask = dpr.layers.LookAheadMask()(tensors, mode)
        mask = tf.maximum(padding_mask, look_ahead_mask)  # (bs, 1, sl, sl)
    else:
        mask = padding_mask  # (bs, 1, 1, sl)

    return mask


class MultiheadAttentionLayer(dpr.layers.Layer):
    """ MultiHead Attention Layer"""

    def __init__(
        self,
        heads: int,
        d_model: int,
        residual_connection: bool,
        emb_dim: int,
        use_layer_normalization: bool,
        block_id: int,
        **kwargs,
    ):
        super().__init__(n_in=2, n_out=1, **kwargs)
        self.heads = heads
        self.d_model = d_model
        self.residual_connection = residual_connection
        self.emb_dim = emb_dim
        self.use_layer_normalization = use_layer_normalization
        self.block_id = block_id

    def forward(self, tensors: Tuple[tf.Tensor, tf.Tensor], mode: str = None):
        with tf.variable_scope(f"multihead_attention_{self.block_id}"):
            input_enc, mask = tensors
            queries = keys = values = input_enc
            batch_size = tf.shape(input_enc)[0]

            Q = tf.layers.dense(
                queries, self.d_model, use_bias=False, trainable=True, kernel_initializer=None
            )  # (bs, sl, d_model)

            K = tf.layers.dense(
                keys, self.d_model, use_bias=False, trainable=True, kernel_initializer=None
            )  # (bs, sl, d_model)

            V = tf.layers.dense(
                values, self.d_model, use_bias=False, trainable=True, kernel_initializer=None
            )  # (bs, sl, d_model)

            Q_ = self.split_heads(Q, batch_size, self.heads, self.d_model)  # (bs, heads, sl, d_model)
            K_ = self.split_heads(K, batch_size, self.heads, self.d_model)  # (bs, heads, sl, d_model)
            V_ = self.split_heads(V, batch_size, self.heads, self.d_model)  # (bs, heads, sl, d_model)

            scaled_attention = self.scaled_dot_product_attention(Q_, K_, V_, mask)  # (bs, heads, sl, d_model/heads)

            # restore shape
            outputs = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (bs, sl, heads, d_model/heads)
            outputs = tf.reshape(outputs, (batch_size, -1, self.d_model))  # (bs, sl, d_model)

            # project the last dim to `emb_dim` size
            outputs = tf.layers.dense(outputs, self.emb_dim)  # (bs, sl, emb_dim)

            # residual connection
            if self.residual_connection:
                outputs += queries

            if self.use_layer_normalization:
                outputs = NormalizationLayer()(outputs, mode)

            return outputs  # (bs, sl, emb_dim)

    @staticmethod
    def split_heads(x, batch_size, heads, d_model):
        """ Split the last dimension into heads """
        x = tf.reshape(x, (batch_size, -1, heads, d_model // heads))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    @staticmethod
    def scaled_dot_product_attention(query, key, value, mask=None):
        """Compute 'Scaled Dot Product Attention'"""
        with tf.variable_scope("scaled_dot_product_attention"):
            d_k = tf.cast(tf.shape(key)[-1], tf.float32)
            scores = tf.matmul(query, key, transpose_b=True, name="DENI_1")
            scores /= tf.math.sqrt(d_k)  # (bs, heads, sl, sl)
            if mask is not None:
                scores += mask * -1e9  # the masked values will be replaced by -1e9
            attention_weights = tf.nn.softmax(scores, axis=-1, name="attention_weights")  # (bs, heads, sl, sl)
            return tf.matmul(attention_weights, value, name="DENI_2")  # (bs, heads, sl, d_model/heads)


def main():
    # 1. load dataset
    train_dataset, validation_dataset = prepare_dataset(
        data_dir="/home/d.kuzin/sync/data/public_datasets/ml-20m",
        min_rating=4,
        min_length=5,
        validation_ratio=0.2,
        nb_negatives=8,
        target_ratio=0.2
    )

    # 2. get input_fns
    train_input_fn = get_input_fn(train_dataset)
    eval_input_fn = get_input_fn(validation_dataset)

    # 3. define prepro_fn
    prepro_fn = get_prepro_fn()

    # 4. define model (pred_fn, loss_fn and optimizer_fn functions)
    pred_fn = TransformerModel(vocab_size=200_000, train_embeddings=True, dim=100)
    loss_fn = dpr.layers.MaskedBPR()
    optimizer_fn = dpr.optimizers.TensorflowOptimizer("Adam", 0.00001)

    # 5. create the train job
    job = dpr.jobs.Trainer(
        path_model="model",
        pred_fn=pred_fn,
        loss_fn=loss_fn,
        optimizer_fn=optimizer_fn,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        prepro_fn=prepro_fn
    )

    job.run()
