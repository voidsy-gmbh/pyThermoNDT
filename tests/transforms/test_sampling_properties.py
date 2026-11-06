"""Tests for sampling invariants using Hypothesis."""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from pythermondt.data import ThermoContainer
from pythermondt.data.units import hertz, second
from pythermondt.transforms import NonUniformSampling, SelectFrameRange, SelectFrames

EXCITATION_SIGNAL = "/MetaData/ExcitationSignal"
TDATA = "/Data/Tdata"
DOMAIN_VALUES = "/MetaData/DomainValues"


@pytest.mark.parametrize("time_domain", [False, True], ids=["frequency", "time"])
@pytest.mark.parametrize("with_excitation", [False, True], ids=["no_excitation", "excitation"])
@pytest.mark.parametrize("select_range", [False, True], ids=["indices", "range"])
@settings(max_examples=50, deadline=None)
@given(
    data=st.data(),
    n_frames=st.integers(1, 48),
    height=st.integers(1, 4),
    width=st.integers(1, 4),
    origin=st.integers(1, 100),
    step=st.sampled_from([0.001, 0.01, 0.1, 1.0, 10.0]),
)
def test_selection_preserves_alignment(
    data, n_frames, height, width, origin, step, time_domain, with_excitation, select_range
):
    """Selection preserves frame identity and resets only the time origin."""
    # Draw bounds from the generated frame count so every selection is valid and non-empty.
    if select_range:
        start = data.draw(st.integers(0, n_frames - 1), label="start")
        end = data.draw(st.integers(start, n_frames - 1), label="end")
        indices = list(range(start, end + 1))
        transform = SelectFrameRange(start, end)
    else:
        # Allow duplicates and reordering: SelectFrames must preserve the requested sequence.
        indices = data.draw(st.lists(st.integers(0, n_frames - 1), min_size=1, max_size=48), label="indices")
        transform = SelectFrames(indices)

    container = ThermoContainer()
    # Unique values expose incorrect frame selection at any spatial location.
    thermal = torch.arange(height * width * n_frames, dtype=torch.float64).reshape(height, width, n_frames)
    domain = origin + step * torch.arange(n_frames, dtype=torch.float64)
    signal = torch.arange(n_frames, dtype=torch.float64) ** 2
    container.update_datasets((TDATA, thermal), (DOMAIN_VALUES, domain))
    unit = second if time_domain else hertz
    container.update_unit(DOMAIN_VALUES, unit)
    if with_excitation:
        container.update_dataset(EXCITATION_SIGNAL, signal)
    else:
        container.remove_dataset(EXCITATION_SIGNAL)

    result = transform(container)

    # Build references frame by frame, independently of bulk tensor indexing.
    expected_thermal = torch.stack([thermal[..., index] for index in indices], dim=-1)
    expected_domain = torch.tensor([origin + step * index for index in indices], dtype=torch.float64)
    if time_domain:
        expected_domain -= origin + step * indices[0]
    assert torch.equal(result.get_dataset(TDATA), expected_thermal)
    assert torch.equal(result.get_dataset(DOMAIN_VALUES), expected_domain)
    assert result.get_unit(DOMAIN_VALUES) == unit
    if with_excitation:
        assert torch.equal(result.get_dataset(EXCITATION_SIGNAL), torch.tensor(indices, dtype=torch.float64) ** 2)
    else:
        assert EXCITATION_SIGNAL not in result.get_all_dataset_paths()


# PyTorch initialization and thread scheduling are not performance assertions.
@settings(max_examples=100, deadline=None)
@given(
    data=st.data(),
    n_frames=st.integers(2, 64),
    height=st.integers(1, 4),
    width=st.integers(1, 4),
    step=st.sampled_from([0.001, 0.01, 0.1, 1.0]),
    slope=st.integers(-20, 20),
    offset=st.integers(-100, 100),
    tau=st.sampled_from([0.01, 0.1, 1.0, 10.0]),
)
def test_linear_sampling_preserves_affine_signals(data, n_frames, height, width, step, slope, offset, tau):
    """Linear interpolation must reproduce a*t+b at every returned time."""
    # Single-sample behavior is covered by the explicit NaN regression test.
    n_samples = data.draw(st.integers(2, n_frames), label="n_samples")
    domain = torch.arange(n_frames, dtype=torch.float64) * step
    # Distinct pixel offsets expose mixing of spatial locations during interpolation.
    spatial_offset = torch.arange(height * width, dtype=torch.float64).reshape(height, width, 1)
    thermal = slope * domain + offset + spatial_offset
    signal = -2 * domain + 3
    container = ThermoContainer()
    container.update_datasets((TDATA, thermal), (DOMAIN_VALUES, domain), (EXCITATION_SIGNAL, signal))

    result = NonUniformSampling(n_samples, tau=tau, interpolate="linear")(container)

    sampled_domain = result.get_dataset(DOMAIN_VALUES)
    assert sampled_domain.shape == (n_samples,)
    assert result.get_dataset(TDATA).shape == (height, width, n_samples)
    assert torch.all(sampled_domain[1:] > sampled_domain[:-1])
    # Float64 roundoff can put the analytically exact endpoint just outside the interval.
    assert sampled_domain.min().item() >= -1e-12
    assert sampled_domain.max().item() <= domain[-1].item() + 1e-12
    torch.testing.assert_close(sampled_domain[[0, -1]], domain[[0, -1]], rtol=1e-12, atol=1e-12)
    # Evaluate the analytical signals at returned times, without repeating the interpolation algorithm.
    torch.testing.assert_close(
        result.get_dataset(TDATA), slope * sampled_domain + offset + spatial_offset, rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(result.get_dataset(EXCITATION_SIGNAL), -2 * sampled_domain + 3, rtol=1e-12, atol=1e-12)
