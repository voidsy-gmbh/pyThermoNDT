import json

import mat73
import numpy as np

from ...data import DataContainer, ThermoContainer
from ...utils import IOPathWrapper
from .base_parser import BaseParser


class SimulationParser(BaseParser):
    supported_extensions = (".mat",)

    @staticmethod
    def parse(data: IOPathWrapper) -> DataContainer:
        """Parses the data from the given IOPathWrapper object into a DataContainer object.

        The IOPathWrapper object must contain a .mat file with simulattion data from COMSOL.

        Parameters:
            data (IOPathWrapper): IOPathWrapper object containing the data to be parsed.

        Returns:
            DataContainer: The parsed data as a DataContainer object.

        Raises:
            ValueError: If the given IOPathWrapper object is empty or does not contain a valid .mat file.
        """
        # Check if the IOPathWrapper object is empty
        if data.file_obj.getbuffer().nbytes == 0:
            raise ValueError("The given IOPathWrapper object is empty.")

        # Try to load the .mat file using mat73 ==> If it fails the file is not a valid .mat file
        try:
            data_dict = mat73.loadmat(data.file_obj, use_attrdict=True)["SimResult"]
        except TypeError as o:
            raise ValueError("The given IOPathWrapper object does not contain a valid .mat file.") from o

        # Create an empty Thermocontainer ==> predefined structure
        datacontainer = ThermoContainer()

        # Add source as an attribute
        datacontainer.add_attributes(path="/MetaData", Source="Simulation")

        # Iterate through all keys and save the values in the datacontainer ==>
        # If one key does not exist the variable in the datapoint will stay None
        for key in data_dict.keys():
            match key:
                case "Tdata":
                    datacontainer.update_dataset(path="/Data/Tdata", data=data_dict[key])
                case "GroundTruth":
                    # Try to extract the label ids and convert to dict
                    try:
                        label_ids = data_dict[key].LabelIds
                        label_ids = json.loads(label_ids) if isinstance(label_ids, str) else label_ids
                        datacontainer.add_attributes(path="/GroundTruth/DefectMask", LabelIds=label_ids)
                    except (json.JSONDecodeError, AttributeError, KeyError):
                        pass
                    datacontainer.update_dataset(path="/GroundTruth/DefectMask", data=data_dict[key]["DefectMask"])
                case "Time":
                    datacontainer.update_dataset(path="/MetaData/DomainValues", data=data_dict[key])
                case "LookUpTable":
                    datacontainer.update_dataset(path="/MetaData/LookUpTable", data=data_dict[key])
                case "ExcitationSignal":
                    datacontainer.update_dataset(path="/MetaData/ExcitationSignal", data=data_dict[key])
                case "ComsolParameters":
                    # Convert Comsol Parameters to a json string
                    converted_comsol_parameters = [
                        [item.item() if isinstance(item, np.ndarray) else item for item in sublist]
                        for sublist in data_dict[key]
                    ]

                    # TODO: Actually this is a list of lists. Should be improved in the future
                    # (maybe with a pandas dataframe ==> needs more work!)
                    # Replace ' with " to make it a valid json string
                    converted_comsol_parameters = str(converted_comsol_parameters).replace("'", '"')

                    # Replace nan with NaN to make it a valid json string
                    converted_comsol_parameters = converted_comsol_parameters.replace("nan", "NaN")

                    # Try to load the json string into a python list
                    try:
                        sim_par = json.loads(converted_comsol_parameters)
                    # If it fails just save the raw string
                    except json.JSONDecodeError:
                        sim_par = converted_comsol_parameters
                    datacontainer.add_attributes(path="/MetaData", SimulationParameter=sim_par)

                case "NoiseLevel":
                    datacontainer.add_attributes(path="/Data/Tdata", NoiseLevel=data_dict[key])
                case "Shapes":
                    # Convert the Shapes into a Python dict first:
                    shapes = {
                        data_dict[key]["Names"][i]: int(data_dict[key]["Numbers"][i])
                        for i in range(len(data_dict[key]["Names"]))
                    }
                    datacontainer.add_attributes(path="/MetaData", Shapes=shapes)

        # Return the constructed datapoint
        return datacontainer
