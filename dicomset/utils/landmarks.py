import numpy as np
import pandas as pd
from typing import List

from ..typing import Landmark, LandmarkID, Landmarks, Point, Points
from .args import arg_to_list

def landmarks_dim(
    landmarks: Landmark | Landmarks,
    ) -> int:
    if isinstance(landmarks, pd.DataFrame):
        axes = [c for c in landmarks.columns if isinstance(c, int)]
    elif isinstance(landmarks, pd.Series):
        axes = [c for c in landmarks.index if isinstance(c, int)]
    else:
        raise ValueError(f"Landmarks must be a pandas DataFrame or Series, got {type(landmarks)}.")
    return len(axes)

def landmarks_to_points(
    landmarks: Landmarks,
    ) -> Points:
    dim = landmarks_dim(landmarks)
    # Need the 'astype' because pd.Series will have mixed types (e.g. landmark-id vs. axes).
    return landmarks[list(range(dim))].astype(np.float32).to_numpy()

def points_to_landmarks(
    points: Point | Points,
    landmark_id: LandmarkID | List[LandmarkID],
    ) -> Landmark | Landmarks:
    if isinstance(points, tuple):
        points = np.array([points])
    dim = points.shape[1]
    assert points.ndim == 2 and dim in (2, 3), f"Points must be of shape (n_landmarks, 2/3) or (2/3,), got {points.shape}."
    landmark_ids = arg_to_list(landmark_id, str)
    assert len(landmark_ids) == len(points), f"Number of landmark IDs must match number of points. Got {len(landmark_ids)} landmark IDs and {len(points)} points."

    # Create series
    if len(landmark_ids) == 1:
        return pd.Series(points.flatten(), index=landmark_ids)

    # Create dataframe.
    landmark_ids = np.array(landmark_ids)[:, np.newaxis]
    data = np.concatenate([landmark_ids, points], axis=1)
    columns = ['landmark-id', *list(range(dim))]
    dtypes = dict((c, np.float32) for c in list(range(dim)))
    df = pd.DataFrame(data, columns=columns).astype(dtypes)
    return df
