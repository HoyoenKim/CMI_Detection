import numpy as np
import polars as pl
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def post_process_for_seg(keys, preds, score_th = 0.01, distance = 5000):
    """make submission dataframe for segmentation task

    Args:
        keys (list[str]): list of keys. key is "{series_id}_{chunk_id}"
        preds (np.ndarray): (num_series * num_chunks, duration, 2)
        score_th (float, optional): threshold for score. Defaults to 0.5.

    Returns:
        pl.DataFrame: submission dataframe
    """
    series_ids = np.array(list(map(lambda x: x.split("_")[0], keys)))
    unique_series_ids = np.unique(series_ids)

    records = []
    plt.figure(figsize=(12, 6))
    for series_id in unique_series_ids:
        series_idx = np.where(series_ids == series_id)[0]
        this_series_preds = preds[series_idx].reshape(-1, 2)

        for i, event_name in enumerate(["onset", "wakeup"]):
            this_event_preds = this_series_preds[:, i]
            steps = find_peaks(this_event_preds, height=score_th, distance=distance)[0]
            scores = this_event_preds[steps]
            
            plt.subplot(2, 1, i + 1)
            plt.plot(this_event_preds, label=f"{event_name} scores")
            plt.scatter(steps, scores, color='red', label='Detected Peaks')
            plt.scatter(np.setdiff1d(np.arange(len(this_event_preds)), steps), this_event_preds[np.setdiff1d(np.arange(len(this_event_preds)), steps)], color='blue', label='Non-peaks')
            plt.title(f"Series ID: {series_id}, Event: {event_name}")
            plt.xlabel("Step")
            plt.ylabel("Score")
            plt.legend()

            for step, score in zip(steps, scores):
                records.append(
                    {
                        "series_id": series_id,
                        "step": step,
                        "event": event_name,
                        "score": score,
                    }
                )
        plt.tight_layout()
        plt.show()

    if len(records) == 0:
        records.append(
            {
                "series_id": series_id,
                "step": 0,
                "event": "onset",
                "score": 0,
            }
        )

    sub_df = pl.DataFrame(records).sort(by=["series_id", "step"])
    row_ids = pl.Series(name="row_id", values=np.arange(len(sub_df)))
    sub_df = sub_df.with_columns(row_ids).select(["row_id", "series_id", "step", "event", "score"])
    return sub_df
