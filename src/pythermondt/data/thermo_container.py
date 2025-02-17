from .datacontainer import DataContainer
from .units import Units


class ThermoContainer(DataContainer):
    def __init__(self):
        """Initializes a DataContainer with a predefined structure for thermographic data"""
        super().__init__()

        # Set initial groups
        self.add_group("/", "Data")
        self.add_group("/", "GroundTruth")
        self.add_group("/", "MetaData")

        # Set initial datasets
        self.add_dataset("/Data", "Tdata")
        self.add_dataset("/GroundTruth", "DefectMask")
        self.add_dataset("/MetaData", "LookUpTable")
        self.add_dataset("/MetaData", "ExcitationSignal")
        self.add_dataset("/MetaData", "DomainValues")

        # Add units to the initial datasets
        self.add_attributes(
            "/Data/Tdata", Unit=Units.arbitrary
        )  # Tdata has a arbitrary unit because it is raw data and the LUT has not been applied yet
        self.add_attributes("/MetaData/DomainValues", Unit=Units.second)
        self.add_attributes("/MetaData/LookUpTable", Unit=Units.kelvin)
        self.add_attributes("/MetaData/ExcitationSignal", Unit=Units.dimensionless)
        self.add_attributes("/GroundTruth/DefectMask", Unit=Units.dimensionless)
