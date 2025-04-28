import json

import numpy as np
import pymatreader
from scipy.io.matlab import MatReadError

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
            data_dict = pymatreader.read_mat(data.file_path)["SimResult"]
        except MatReadError as o:
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
                        label_ids = data_dict[key]["LabelIds"]
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
                    # Reshape comsol parameters back to a list of lists ==> pymatreader loads this as a flattened list
                    flattenend_comsol_parameters = reshape_pymatreader_parameters(data_dict[key])

                    # Convert Comsol Parameters to a json string
                    converted_comsol_parameters = [
                        [item.item() if isinstance(item, np.ndarray) and item.size == 1 else item for item in sublist]
                        for sublist in flattenend_comsol_parameters
                    ]

                    # Clean up the list of lists ==> handle empty arrays and single dimensions
                    for i, sublist in enumerate(converted_comsol_parameters):
                        for j, item in enumerate(sublist):
                            if isinstance(item, np.ndarray) and item.size == 1:
                                converted_comsol_parameters[i][j] = item.item()
                            if isinstance(item, np.ndarray) and item.size == 0:
                                converted_comsol_parameters[i][j] = ""

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


def reshape_pymatreader_parameters(flat_params):
    # Determine where the sections start
    num_params = 46  # Based on your example

    # Extract each section
    param_names = flat_params[:num_params]
    expressions = flat_params[num_params : 2 * num_params]
    descriptions = flat_params[2 * num_params : 3 * num_params]
    values = flat_params[3 * num_params : 4 * num_params]
    units = flat_params[4 * num_params : 5 * num_params]

    # Create the column headers
    # result = [['Name', 'Expression', 'Description', 'Value', 'Unit']]
    result = []

    # Create a row for each parameter
    for i in range(num_params):
        result.append([param_names[i], expressions[i], descriptions[i], values[i], units[i]])

    return result
