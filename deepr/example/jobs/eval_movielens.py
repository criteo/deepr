"""Eval dataset."""

import logging
from dataclasses import dataclass

import faiss
import numpy as np
import deepr as dpr


LOGGER = logging.getLogger(__name__)


@dataclass
class EvalMovieLens(dpr.jobs.Job):
    """Eval MovieLens."""

    path_predictions: str
    path_embeddings: str
    k: int

    def run(self):
        with dpr.io.ParquetDataset(self.path_predictions).open() as ds:
            predictions = ds.read_pandas().to_pandas()
            users = np.stack(predictions["user"])

        with dpr.io.ParquetDataset(self.path_embeddings).open() as ds:
            embeddings = ds.read_pandas().to_pandas()
            embeddings = embeddings.to_numpy()

        index = faiss.IndexFlatIP(embeddings.shape[-1])
        index.add(np.ascontiguousarray(embeddings))
        _, indices = index.search(users, k=self.k)
        for gold, pred in zip(predictions["target"], indices):
            print(np.mean(gold == pred))

# class ValidationComputer:
#     @utils.check_kwargs_decorator
#     def run(
#         self,
#         item_embeddings_path: str = None,
#         user_embeddings_path: str = None,
#         validation_path: str = None,
#         num_items: List[int] = [10, 20, 50],
#         popularity_threshold: int = 50,
#         metadata_path: str = None,
#         from_hdfs: bool = True,
#         knn_distance: str = "ip",  # possible options are l2, cosine or ip
#         output_path: str = None,
#     ):
#         """
#         The prefered way to run this is for instance:
#             python -m deep_reco.run_validation run \
#                 --item_embeddings_path="./deep_reco/data/ml-20m/fine_tuned_item_embeddings" \
#                 --user_embeddings_path="./deep_reco/data/ml-20m/open_sourcing_2020-02-03-16-57-36/user-embeddings/embeddings/"
#                 --validation_path="./deep_reco/data/ml-20m/validation/tf_records/validation"
#                 --metadata_path="./deep_reco/data/metadata/"
#         """

#         # load metadata
#         metadata_mapping = utils.load_metadata_file_and_create_mapping(
#             metadata_path=metadata_path, read_as_parquet=False, from_hdfs=from_hdfs
#         )
#         popular_items_dict = self.compute_popular_items(
#             metadata_mapping.metadata_df, popularity_threshold
#         )

#         # read validation dataset
#         grouped_validation_set_df = self.load_validation_set(validation_path)

#         # read item and user embeddings
#         (item_embeddings, user_embeddings, embedding_dim) = self.load_embeddings(
#             item_embeddings_path, user_embeddings_path, from_hdfs
#         )

#         # compute indices per group
#         indices = self.compute_indices(item_embeddings, embedding_dim, knn_distance)

#         # enrich validation set with user embeddings
#         validation_df = grouped_validation_set_df.set_index("uid").join(
#             user_embeddings.set_index("uid"), how="inner"
#         )

#         result_dict = {}

#         for num_neighbours in num_items:
#             # compute predicted items
#             validation_df[f"predictedItems@{num_neighbours}"] = validation_df.apply(
#                 lambda row: self.find_nearest_items(
#                     int(row["groupId"]), row["embedding"], indices, num_neighbours
#                 ),
#                 axis=1,
#             )

#             # compute metrics per user/group
#             metrics_df = validation_df.apply(
#                 lambda row: self.compute_all_metrics(
#                     row[f"predictedItems@{num_neighbours}"],
#                     row["validationItemIds"],
#                     popular_items_dict[row["groupId"]][
#                         :num_neighbours
#                     ],  # keep only the first num_neighbours best ofs
#                 ),
#                 axis=1,
#             )

#             # aggregate metrics
#             result = (
#                 pd.DataFrame(metrics_df.tolist(), index=metrics_df.index)
#                 .apply(np.nanmean, axis=0)
#                 .values
#             )
#             result_dict.update(
#                 {
#                     f"precision@{num_neighbours}": result[0],
#                     f"precisionWithoutGBO@{num_neighbours}": result[1],
#                     f"precisionGBO@{num_neighbours}": result[2],
#                     f"recall@{num_neighbours}": result[3],
#                     f"recallWithoutGBO@{num_neighbours}": result[4],
#                     f"recallGBO@{num_neighbours}": result[5],
#                 }
#             )

#         if output_path is None:
#             # infer output path from user embedding path
#             output_path = path.abspath(path.join(user_embeddings_path, "../.."))
#         output_path_metrics = path.join(output_path, "validation_metrics.json")

#         logging.info("metrics = {}".format(result_dict))
#         logging.info(f"Writing metrics to {output_path}")

#         # save to file
#         utils.write_dict_as_json_hdfs(
#             result_dict, output_path_metrics, from_hdfs=from_hdfs
#         )

#     @staticmethod
#     def find_nearest_items(group_id, user_embedding, indices, num_items):
#         """ find nearest items to the user embedding using indices passed as input """
#         if group_id in indices:
#             # Query dataset, k - number of closest elements (returns 2 numpy arrays)
#             labels, _ = indices[group_id].knn_query(user_embedding, k=num_items)
#             return labels[0]
#         else:
#             return None

#     @staticmethod
#     def load_embeddings(item_embeddings_path, user_embeddings_path, from_hdfs):
#         """ load item and user embeddings from local files """
#         logger.info(
#             "reading embeddings from {} and {}".format(
#                 item_embeddings_path, user_embeddings_path
#             )
#         )
#         item_embeddings = utils.read_embeddings(
#             item_embeddings_path, from_hdfs=from_hdfs
#         )
#         user_embeddings = utils.read_embeddings(
#             user_embeddings_path, from_hdfs=from_hdfs
#         )
#         user_embeddings["uid"] = user_embeddings.apply(
#             lambda row: utils.uuid_128b_to_hex(
#                 row["mostSignificantBits"], row["leastSignificantBits"]
#             ),
#             axis=1,
#         )
#         user_embeddings_df = pd.DataFrame(user_embeddings).drop(
#             ["mostSignificantBits", "leastSignificantBits"], axis=1
#         )
#         embedding_size = len(user_embeddings_df["embedding"].iloc[0])
#         return item_embeddings, user_embeddings_df, embedding_size

#     @staticmethod
#     def load_validation_set(path):
#         validation_data: List[Record] = read_tf_records(path)
#         validation_set = []
#         for d in validation_data:
#             for (p, x) in list(zip(d.targetGroupIds, d.targetPositiveItemIds)):
#                 validation_set.append([d.uid, p, x])
#         validation_set_df = pd.DataFrame(validation_set)
#         validation_set_df.columns = ["uid", "groupId", "itemId"]
#         validation_set_df["uid"] = validation_set_df["uid"].apply(
#             lambda uid: uid.decode("ascii")
#         )
#         grouped_validation_set_df = (
#             validation_set_df.groupby(["uid", "groupId"])["itemId"]
#             .apply(list)
#             .reset_index(name="validationItemIds")
#         )
#         return grouped_validation_set_df

#     @staticmethod
#     def compute_indices(item_embeddings, embedding_dim, knn_distance):
#         """ compute knn indices from item embeddings """

#         import hnswlib

#         indices = {}
#         group_ids = item_embeddings["groupId"].to_numpy()
#         for group_id in np.unique(group_ids):
#             logger.info("indexing group {}".format(group_id))
#             group_items_mapping = item_embeddings[group_ids == group_id]

#             # filter group data
#             item_embeddings_2d = np.stack(group_items_mapping["embedding"].to_numpy())
#             max_elements = item_embeddings_2d.shape[0]
#             logger.info(
#                 "num item embeddings for group {} = {}".format(group_id, max_elements)
#             )
#             # fit nearest neighbors
#             p = hnswlib.Index(space=knn_distance, dim=embedding_dim)
#             # Initing index - the maximum number of elements should be known beforehand
#             p.init_index(max_elements=max_elements, ef_construction=200, M=16)
#             item_embeddings_labels = group_items_mapping["itemId"]

#             logger.info("fitting nearest neighbors...")
#             p.add_items(item_embeddings_2d, item_embeddings_labels)
#             # Controlling the recall by setting ef:
#             p.set_ef(50)  # ef should always be > k
#             indices[group_id] = p
#         return indices

#     @staticmethod
#     def compute_popular_items(metadata_df, top_n):
#         """ compute the <top_n> most popular items per group from <metadata_df>
#         which is a dataframe of: groupId  hashedExternalId  globalPopularity  index  ucId  uc  title
#         <globalPopularity> is an absolute count of events for this item
#         """
#         sorted_df = (
#             metadata_df.sort_values(["groupId", "globalPopularity"], ascending=False)
#             .groupby("groupId")
#             .head(top_n)
#         )
#         popular_items_per_group_dict = (
#             sorted_df.groupby("groupId")["hashedExternalId"]
#             .apply(list)
#             .reset_index(name="popularItems")
#             .set_index("groupId")
#             .to_dict()["popularItems"]
#         )

#         x = sorted_df.groupby("groupId")["hashedExternalId", "globalPopularity"].apply(
#             list
#         )
#         return popular_items_per_group_dict

#     @staticmethod
#     def compute_precision(predicted_items, observed_items, popular_items):
#         """ compute recall with and without removal of popular items
#         """
#         # standard recall
#         intersect = list(set(predicted_items) & set(observed_items))
#         num_predicted = len(predicted_items)
#         precision = len(intersect) / num_predicted
#         # unbiased recall
#         observed_items_without_popular = list(set(observed_items) - set(popular_items))
#         intersect_without_popular = list(
#             set(predicted_items) & set(observed_items_without_popular)
#         )
#         precision_without_popular = len(intersect_without_popular) / num_predicted

#         precision_of_popular = (
#             len(set(observed_items) & set(popular_items)) / num_predicted
#         )

#         return precision, precision_without_popular, precision_of_popular

#     @staticmethod
#     def compute_recall(predicted_items, observed_items, popular_items):
#         """ compute recall with and without removal of popular items
#         """
#         # standard recall
#         intersect = list(set(predicted_items) & set(observed_items))
#         num_predicted = len(predicted_items)
#         num_observed = len(set(observed_items))
#         recall = len(intersect) / num_observed
#         # unbiased recall
#         observed_items_without_popular = list(set(observed_items) - set(popular_items))
#         num_observed_without_popular = len(observed_items_without_popular)
#         intersect_without_popular = list(
#             set(predicted_items) & set(observed_items_without_popular)
#         )
#         recall_without_popular = np.divide(
#             len(intersect_without_popular), num_observed_without_popular
#         )  # will give nan if num_observed_without_popular is 0
#         recall_of_popular = len(set(observed_items) & set(popular_items)) / num_observed
#         return recall, recall_without_popular, recall_of_popular

#     @staticmethod
#     def compute_all_metrics(predicted_items, observed_items, popular_items):
#         (
#             precision,
#             precision_without_popular,
#             precision_of_popular,
#         ) = ValidationComputer.compute_precision(
#             predicted_items, observed_items, popular_items
#         )
#         (
#             recall,
#             recall_without_popular,
#             recall_of_popular,
#         ) = ValidationComputer.compute_recall(
#             predicted_items, observed_items, popular_items
#         )
#         return (
#             precision,
#             precision_without_popular,
#             precision_of_popular,
#             recall,
#             recall_without_popular,
#             recall_of_popular,
#         )
