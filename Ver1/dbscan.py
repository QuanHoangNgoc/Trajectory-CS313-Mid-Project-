from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import data
from data import *


def extract_feature_vector(traj):
    assert len(traj) >= 2

    vectors = []
    for i in range(1, len(traj)):
        direct = traj[i] - traj[i - 1]
        lnorm = np.linalg.norm(direct)
        if lnorm != 0:
            direct = direct / lnorm
        else:
            direct = np.array([0, 0])

        feat = np.array([traj[i - 1], traj[i], direct]).flatten()  #!!! flatten
        feat = np.append(feat, lnorm)
        vectors.append(feat)
        assert len(feat) == 7

    scaler = StandardScaler()  #!!! Scale features before KNN
    vectors = scaler.fit_transform(vectors)
    return np.array(vectors)


def to_vectors(data) -> np.array:
    vectors = None
    for traj in data:
        if len(traj) < 2:
            continue
        if vectors is None:
            vectors = extract_feature_vector(traj)
        else:
            tmp = extract_feature_vector(traj)
            vectors = np.concatenate((vectors, tmp), axis=0)

    return vectors


def vectors_to_labels(vectors, C, min_samples) -> tuple:
    # Create a DBSCAN object with parameters
    dbscan = DBSCAN(
        eps=C, min_samples=min_samples
    )  # Adjust eps and min_samples as needed
    dbscan.fit(vectors)
    labels = dbscan.labels_
    # Identify noise points (cluster -1)
    noise_indices = np.where(labels == -1)[0]
    return labels, noise_indices


def to_segments(data) -> np.array:
    segments = []
    for traj in data:
        if len(traj) < 2:
            continue
        for i in range(1, len(traj)):
            seg = np.array([traj[i - 1], traj[i]])
            segments.append(seg)
    return np.array(segments)


def get_segments_with_clusid(clusid, segments, labels) -> np.array:
    ids = np.where(labels == clusid)[0].tolist()
    sub_segments = [segments[i] for i in ids]  # Use list comprehension
    return sub_segments


def test_cluster(segments, labels, noise_indices):
    print(f"# of segments: {len(segments)}")
    print(f"# of labels: {len(labels)}")
    print(f"# of unique clusters: {max(labels)}")
    for i in range(max(labels)):
        print(
            f"# of segments in cluster {i}: {len(get_segments_with_clusid(i, segments, labels))}"
        )
    print(f"# of noise indices: {len(noise_indices)}")
