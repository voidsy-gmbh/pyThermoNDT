"""Shared fixtures for transform tests."""

import pytest
import torch

from pythermondt.data import ThermoContainer
from pythermondt.data.units import second

from .utils import ThermoSequenceFactory


@pytest.fixture
def thermo_sequence() -> ThermoSequenceFactory:
    """Factory for a minimal time-domain ThermoContainer.

    Builds ``Tdata`` (H x W x T), uniformly spaced ``DomainValues``, and optionally
    an ``ExcitationSignal``. Suitable as a base for sampling and frequency transforms.

    Example:
        container = thermo_sequence(n_frames=32)
        container = thermo_sequence(n_frames=20, excitation_signal=False)
        container = thermo_sequence(n_frames=20, excitation_signal=custom_signal)
    """

    def _factory(
        n_frames: int = 32,
        *,
        height: int = 4,
        width: int = 5,
        excitation_signal: bool | torch.Tensor = True,
        dt: float = 0.01,
        seed: int = 0,
    ) -> ThermoContainer:
        container = ThermoContainer()
        generator = torch.Generator().manual_seed(seed)
        tdata = torch.randn(height, width, n_frames, dtype=torch.float64, generator=generator)
        domain_values = torch.arange(n_frames, dtype=torch.float64) * dt

        container.update_dataset("/Data/Tdata", tdata)
        container.update_dataset("/MetaData/DomainValues", domain_values)
        container.update_unit("/MetaData/DomainValues", second)

        if excitation_signal is False:
            container.remove_dataset("/MetaData/ExcitationSignal")
        elif excitation_signal is True:
            signal = torch.zeros(n_frames, dtype=torch.float64)
            signal[:3] = 1.0
            container.update_dataset("/MetaData/ExcitationSignal", signal)
        else:
            if excitation_signal.shape[-1] != n_frames:
                raise ValueError(
                    f"excitation_signal length {excitation_signal.shape[-1]} does not match n_frames={n_frames}."
                )
            container.update_dataset("/MetaData/ExcitationSignal", excitation_signal.to(dtype=torch.float64))

        return container

    return _factory
