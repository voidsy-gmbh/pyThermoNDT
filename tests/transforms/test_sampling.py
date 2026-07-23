"""Tests for sampling transforms (SelectFrames, SelectFrameRange, ...)."""

import pytest
import torch

from pythermondt.data.units import hertz
from pythermondt.transforms import Compose, ExtractPhase, PulsePhaseThermography, SelectFrameRange, SelectFrames

from .utils import ThermoSequenceFactory, has_dataset

EXCITATION_SIGNAL = "/MetaData/ExcitationSignal"
TDATA = "/Data/Tdata"
DOMAIN_VALUES = "/MetaData/DomainValues"


@pytest.mark.parametrize(
    ("transform", "n_frames", "expected_len"),
    [
        (SelectFrameRange(start=2, end=10), 20, 9),
        (SelectFrames([0, 5, 10]), 20, 3),
    ],
    ids=["frame_range", "frames"],
)
def test_select_without_excitation_signal(
    thermo_sequence: ThermoSequenceFactory,
    transform: SelectFrameRange | SelectFrames,
    n_frames: int,
    expected_len: int,
):
    """Select* must work when ExcitationSignal is absent."""
    container = thermo_sequence(n_frames=n_frames, excitation_signal=False)

    result = transform(container)

    assert result.get_dataset(TDATA).shape[-1] == expected_len
    assert result.get_dataset(DOMAIN_VALUES).shape[-1] == expected_len
    assert not has_dataset(result, EXCITATION_SIGNAL)


@pytest.mark.parametrize(
    ("transform", "n_frames", "expected_signal"),
    [
        (SelectFrames([0, 5, 10]), 20, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)),
        (SelectFrameRange(start=0, end=4), 20, torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0], dtype=torch.float64)),
    ],
    ids=["frames", "frame_range"],
)
def test_select_keeps_excitation_signal_in_sync(
    thermo_sequence: ThermoSequenceFactory,
    transform: SelectFrameRange | SelectFrames,
    n_frames: int,
    expected_signal: torch.Tensor,
):
    """When present, ExcitationSignal is sliced together with Tdata."""
    container = thermo_sequence(n_frames=n_frames, excitation_signal=True)

    result = transform(container)

    signal = result.get_dataset(EXCITATION_SIGNAL)
    assert signal.shape[-1] == expected_signal.shape[-1]
    assert result.get_dataset(TDATA).shape[-1] == expected_signal.shape[-1]
    assert torch.equal(signal, expected_signal)


def test_select_frame_range_zero_bases_time_domain(thermo_sequence: ThermoSequenceFactory):
    """Time-domain DomainValues are zero-based after a crop that does not start at 0."""
    container = thermo_sequence(n_frames=20, excitation_signal=False, dt=0.01)

    result = SelectFrameRange(start=2, end=10)(container)

    domain = result.get_dataset(DOMAIN_VALUES)
    assert torch.allclose(domain[0], torch.tensor(0.0, dtype=domain.dtype))
    assert torch.allclose(domain[1] - domain[0], torch.tensor(0.01, dtype=domain.dtype))


def test_select_frame_range_keeps_absolute_frequency_domain(thermo_sequence: ThermoSequenceFactory):
    """Frequency-domain DomainValues keep absolute Hz (no zero-base after PPT)."""
    container = PulsePhaseThermography()(thermo_sequence(n_frames=64))
    freqs_before = container.get_dataset(DOMAIN_VALUES).clone()
    assert container.get_unit(DOMAIN_VALUES) == hertz

    start, end = 5, 12
    result = SelectFrameRange(start=start, end=end)(container)

    freqs_after = result.get_dataset(DOMAIN_VALUES)
    assert torch.allclose(freqs_after, freqs_before[start : end + 1])
    assert torch.allclose(freqs_after[0], freqs_before[start])


def test_select_frame_range_after_ppt_and_extract_phase(thermo_sequence: ThermoSequenceFactory):
    """Regression: PPT drops ExcitationSignal; SelectFrameRange must still run.

    https://github.com/voidsy-gmbh/pyThermoNDT/issues/458
    """
    end = 32
    result = Compose(
        [
            PulsePhaseThermography(),
            ExtractPhase(),
            SelectFrameRange(end=end),
        ]
    )(thermo_sequence(n_frames=64))

    n_keep = end + 1  # end is inclusive
    assert result.get_dataset(TDATA).shape[-1] == n_keep
    assert result.get_dataset(DOMAIN_VALUES).shape[-1] == n_keep
    assert not has_dataset(result, EXCITATION_SIGNAL)
