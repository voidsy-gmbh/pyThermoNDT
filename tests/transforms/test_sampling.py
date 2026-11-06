"""Regression and transform-chain tests for sampling; general invariants use Hypothesis."""

import pytest
import torch

from pythermondt.data.units import hertz
from pythermondt.transforms import Compose, ExtractPhase, NonUniformSampling, PulsePhaseThermography, SelectFrameRange

from .utils import ThermoSequenceFactory, has_dataset

EXCITATION_SIGNAL = "/MetaData/ExcitationSignal"
TDATA = "/Data/Tdata"
DOMAIN_VALUES = "/MetaData/DomainValues"


@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason="NonUniformSampling accepts n_samples=1 but returns NaNs."
)
def test_single_sample_is_finite(thermo_sequence: ThermoSequenceFactory):
    """A sample count within the advertised bounds must not introduce NaNs."""
    container = thermo_sequence(n_frames=2)

    result = NonUniformSampling(1, tau=1.0, interpolate="linear")(container)

    for path in (TDATA, DOMAIN_VALUES, EXCITATION_SIGNAL):
        assert torch.isfinite(result.get_dataset(path)).all()


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
