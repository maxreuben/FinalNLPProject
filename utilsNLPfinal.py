import collections
import json
import logging
import os
from typing import Optional, Tuple

import numpy as np
from tqdm.auto import tqdm


logger = logging.getLogger(__name__)


def postprocesspredictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    version_2_with_negative: bool = False,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    null_score_diff_threshold: float = 0.0,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    log_level: Optional[int] = logging.WARNING,
):

    assert len(predictions) == 2, "`predictions` should be of length 2 tuples"
    start_logits, end_logits_cummi = predictions

    assert len(predictions[0]) == len(features), f"Got {len(predictions[0])} predictions and {len(features)} features."
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for imo, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(imo)
    predictions_all = collections.OrderedDict()
    json_nbest = collections.OrderedDict()
    if not version_2_with_negative:
        pass
    else:
        scores_diff_json = collections.OrderedDict()
    logger.setLevel(log_level)
    logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")
    helper_function_process(end_logits_cummi, json_nbest, predictions_all, start_logits, examples, features,
                            features_per_example, max_answer_length, n_best_size, null_score_diff_threshold,
                            scores_diff_json, version_2_with_negative)

    if output_dir is None:
        pass
    else:
        assert os.path.isdir(output_dir), f"{output_dir} is not a directory."

        prediction_file = os.path.join(
            output_dir, "predictions.json" if prefix is None else f"{prefix}_predictions.json"
        )
        nbest_file = os.path.join(
            output_dir, "nbest_predictions.json" if prefix is None else f"{prefix}_nbest_predictions.json"
        )
        if not version_2_with_negative:
            pass
        else:
            null_odds_file = os.path.join(
                output_dir, "null_odds.json" if prefix is None else f"{prefix}_null_odds.json"
            )

        logger.info(f"Saving predictions to {prediction_file}.")
        with open(prediction_file, "w") as writer:
            writer.write(json.dumps(predictions_all, indent=4) + "\n")
        logger.info(f"Saving nbest_preds to {nbest_file}.")
        with open(nbest_file, "w") as writer:
            writer.write(json.dumps(json_nbest, indent=4) + "\n")
        if not version_2_with_negative:
            return
        logger.info(f"Saving null_odds to {null_odds_file}.")
        with open(null_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return predictions_all


def helper_function_process(end_logits_cummi, all_nbest_json, all_predictions, start_logits, examples, features,
                            features_per_example, max_answer_length, n_best_size, null_score_diff_threshold,
                            scores_diff_json, version_2_with_negative):
    for index_examp, jumpexamp in enumerate(tqdm(examples)):
        feature_indices = features_per_example[index_examp]

        min_null_prediction = None
        prelim_predictions = []
        min_null_prediction = helper_function(end_logits_cummi, start_logits, feature_indices, features,
                                              max_answer_length, min_null_prediction, n_best_size, prelim_predictions)
        if not version_2_with_negative:
            pass
        else:
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction["score"]

        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

        if not version_2_with_negative or any(p["offsets"] == (0, 0) for p in predictions):
            pass
        else:
            predictions.append(min_null_prediction)

        context = jumpexamp["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0]: offsets[1]]

        if len(predictions) != 0 and (len(predictions) != 1 or predictions[0]["text"] != ""):
            pass
        else:
            predictions.insert(0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})

        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        if version_2_with_negative:
            i = 0
            while predictions[i]["text"] == "":
                i += 1
            best_non_null_pred = predictions[i]

            score_diff = null_score - best_non_null_pred["start_logit"] - best_non_null_pred["end_logit"]
            scores_diff_json[jumpexamp["id"]] = float(score_diff)  # To be JSON-serializable.
            if score_diff <= null_score_diff_threshold:
                all_predictions[jumpexamp["id"]] = best_non_null_pred["text"]
            else:
                all_predictions[jumpexamp["id"]] = ""
        else:
            all_predictions[jumpexamp["id"]] = predictions[0]["text"]

        all_nbest_json[jumpexamp["id"]] = [
            {k: (v if not isinstance(v, (np.float16, np.float32, np.float64)) else float(v)) for k, v in pred.items()}
            for pred in predictions
        ]


def helper_function(end_logits_cummi, start_logits, feature_indices, features, max_answer_length, min_null_prediction,
                    n_best_size, prelim_predictions):
    for indexesofthefeatures in feature_indices:
        logits_starter = start_logits[indexesofthefeatures]
        logits_ender = end_logits_cummi[indexesofthefeatures]
        mapping_of_offset = features[indexesofthefeatures]["offset_mapping"]
        maxcontentofthetoken = features[indexesofthefeatures].get("token_is_max_context", None)
        feature_null_score = logits_starter[0] + logits_ender[0]
        if min_null_prediction is not None and min_null_prediction["score"] <= feature_null_score:
            pass
        else:
            min_null_prediction = {
                "offsets": (0, 0),
                "score": feature_null_score,
                "start_logit": logits_starter[0],
                "end_logit": logits_ender[0],
            }
        start_indexes = np.argsort(logits_starter)[-1: -n_best_size - 1: -1].tolist()
        end_indexes = np.argsort(logits_ender)[-1: -n_best_size - 1: -1].tolist()
        for starterindex in start_indexes:
            for end in end_indexes:

                if starterindex < len(mapping_of_offset) and end < len(mapping_of_offset) and mapping_of_offset[
                    starterindex] is not None and \
                        mapping_of_offset[end] is not None:
                    if end < starterindex or end - starterindex + 1 > max_answer_length:
                        continue

                    if maxcontentofthetoken is not None and not maxcontentofthetoken.get(str(starterindex), False):
                        continue
                    prelim_predictions.append(
                        {
                            "offsets": (mapping_of_offset[starterindex][0], mapping_of_offset[end][1]),
                            "score": logits_starter[starterindex] + logits_ender[end],
                            "start_logit": logits_starter[starterindex],
                            "end_logit": logits_ender[end],
                        }
                    )
    return min_null_prediction