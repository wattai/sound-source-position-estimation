# -*- coding: utf-8 -*-

import numpy as np

from sse.music_v2 import MusicSoundSourceLocator


class TestMusicSoundSourceLocator:
    def test_fit_transform(self):
        self.locator = MusicSoundSourceLocator(
            fs=16000,
            d=1.0,
        )
        X = np.random.randn(1000, 2)
        out = self.locator.fit_transform(X=X)
        print(out)
