# BSD License
#
# For fairmotion software
#
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Modified by Ruilong Li
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the conditions and disclaimer in utils.py are retained.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS".

import numpy as np

from . import utils as feat_utils


def extract_kinetic_features(positions):
    assert len(positions.shape) == 3
    features = KineticFeatures(positions)
    kinetic_feature_vector = []
    for i in range(positions.shape[1]):
        feature_vector = np.hstack(
            [
                features.average_kinetic_energy_horizontal(i),
                features.average_kinetic_energy_vertical(i),
                features.average_energy_expenditure(i),
            ]
        )
        kinetic_feature_vector.extend(feature_vector)
    return np.array(kinetic_feature_vector, dtype=np.float32)


class KineticFeatures:
    def __init__(self, positions, frame_time=1.0 / 60, up_vec="y", sliding_window=2):
        self.positions = positions
        self.frame_time = frame_time
        self.up_vec = up_vec
        self.sliding_window = sliding_window

    def average_kinetic_energy_horizontal(self, joint):
        value = 0
        for i in range(1, len(self.positions)):
            velocity = feat_utils.calc_average_velocity_horizontal(
                self.positions, i, joint, self.sliding_window, self.frame_time, self.up_vec
            )
            value += velocity**2
        return value / (len(self.positions) - 1.0)

    def average_kinetic_energy_vertical(self, joint):
        value = 0
        for i in range(1, len(self.positions)):
            velocity = feat_utils.calc_average_velocity_vertical(
                self.positions, i, joint, self.sliding_window, self.frame_time, self.up_vec
            )
            value += velocity**2
        return value / (len(self.positions) - 1.0)

    def average_energy_expenditure(self, joint):
        value = 0.0
        for i in range(1, len(self.positions)):
            value += feat_utils.calc_average_acceleration(
                self.positions, i, joint, self.sliding_window, self.frame_time
            )
        return value / (len(self.positions) - 1.0)

