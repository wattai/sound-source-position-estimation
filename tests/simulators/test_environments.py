import numpy as np
import pytest

from sse.simulators.environments import (
    Observer,
    Air,
    Microphone,
    Source,
    Position3D,
    SineSignalGenerator,
    calc_distance,
)


@pytest.mark.parametrize(
    "source, microphone",
    [
        (
            Source(
                position=Position3D(r=50, theta=np.pi, phi=np.pi / 2),
                signal=SineSignalGenerator(frequency=6000).generate(
                    sampling_frequency=16000,
                    time_length=1,
                ),
            ),
            Microphone(
                position=Position3D(r=100, theta=-0.5, phi=1.1),
                sampling_frequency=16000,
            ),
        ),
    ],
)
def test_num_delayed_points_of_signal(source: Source, microphone: Microphone):
    medium = Air()
    obs = Observer(
        sources=[source],
        microphones=[microphone],
        medium=medium,
    )
    outs = obs.ring_sources()
    shift = find_max_correlation_shift(
        signal1=obs.sources[0].signal.values,
        signal2=outs[0].values,
    )
    print(f"相互相関が最大となるシフト量: {shift}")
    assert shift == int(
        round(
            calc_distance(source.position, microphone.position)
            / medium.sound_speed
            * microphone.sampling_frequency
        )
    )


def find_max_correlation_shift(signal1, signal2):
    """
    2つの信号の相互相関が最大となる位置を返す関数
    Args:
        signal1 (array-like): 最初の信号配列
        signal2 (array-like): 2番目の信号配列

    Returns:
        int: 相互相関が最大になるときの信号2のシフト量
    """
    # 長さを揃えるためのゼロパディング
    len(signal1) + len(signal2) - 1
    corr = np.correlate(signal1, signal2, mode="full")

    # 最大相互相関値のインデックスを見つける
    max_corr_index = np.argmax(corr)

    # ズレ位置を計算（中央を基準としてラグを計算）
    shift = max_corr_index - (len(signal2) - 1)
    return shift
