# vim: expandtab:ts=4:sw=4
from copy import deepcopy

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, cat, mean, covariance, track_id, n_init, max_age,
                 feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.cat = cat

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age
        self._real_max_age = max_age

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf, fov=None):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        fov : the tuple of width and height of the frame
        """
        old_mean = self.mean
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1
        self.mean[2] = old_mean[2] * 0.9 + self.mean[2] * 0.1
        self.mean[3] = old_mean[3] * 0.9 + self.mean[3] * 0.1

        # check if the predict is out fo the fov. if so, reduce the real max age
        if fov is not None:
            if self.mean[0] < 0 or self.mean[0] > fov[1] or self.mean[1] < 0 or self.mean[1] > fov[0]:
                self._real_max_age = min(30, int(self._max_age / 2))

    def update(self, kf, detection, shape):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        shape: frame size
        """
        boundary = 20
        (h, w) = shape
        # det = deepcopy(detection)
        det = detection
        # if track is person and near the boundary
        if self.cat == 1 and self.hits > 10 and det.tlwh[0] > boundary and det.tlwh[1] > boundary \
                and det.tlwh[0] + det.tlwh[2] < w - boundary and det.tlwh[1] + det.tlwh[3] < h - boundary:
            det_w = detection.tlwh[2]
            det_h = detection.tlwh[3]
            self_w = self.mean[3] * self.mean[2]
            self_h = self.mean[3]
            diff_w = abs(self_w - det_w) / self_w
            diff_h = abs(self_h - det_h) / self_h

            # if the size change too much, not updating
            if self.time_since_update == 1 and diff_h > 0.2 and diff_w > 0.2:
                return False

            d_xyah = detection.to_xyah()
            cx = d_xyah[0]
            cy = d_xyah[1]

            # make the aspect ratio similar to the tracking box
            if diff_w < diff_h:
                det_h = det_h * 0.1 + self_h * 0.9
            else:
                det_w = det_w * 0.5 + self_w * 0.5

            det.tlwh[0] = cx - 0.5 * det_w
            det.tlwh[1] = cy - 0.5 * det_h
            det.tlwh[2] = det_w
            det.tlwh[3] = det_h

        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, det.to_xyah())

        self.hits += 1
        self._real_max_age = min(self.hits + 10 - self._n_init, self._max_age)
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed
        return True

    def append_feature(self, feature):
        self.features.append(feature)

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._real_max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
