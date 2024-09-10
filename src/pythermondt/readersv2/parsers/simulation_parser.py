import io
import mat73
import json
import numpy as np
from .base_parser import BaseParser
from ...data import DataContainer, ThermoContainer

class SimulationParser(BaseParser):
    def parse(self, data_bytes: io.BytesIO) -> DataContainer:
        """ Parses the data from the given BytesIO object, that was read using one of the BaseReaders subclasses into a DataContainer object.

        The BytesIO object must contain a .mat file with simulattion data from COMSOL.

        Parameters:
            data_bytes (io.BytesIO): The BytesIO object containing the data to be parsed.

        Returns:
            DataContainer: The parsed data as a DataContainer object.

        Raises:
            ValueError: If the given BytesIO object is empty or does not contain a valid .mat file.
        """
        # Check if the BytesIO object is empty
        if data_bytes.getbuffer().nbytes == 0:
            raise ValueError("The given BytesIO object is empty.")
        
        # Try to load the .mat file using mat73 ==> If it fails the file is not a valid .mat file
        try:
            data = mat73.loadmat(data_bytes, use_attrdict=True)['SimResult']
        except OSError:
            raise ValueError("The given BytesIO object does not contain a valid .mat file.")
        
        # Create an empty DataContainer
        datacontainer = ThermoContainer()

        # Add source as an attribute
        datacontainer.add_attributes(path='/MetaData', Source='Simulation')

        # Domaintype will always be time in case of the Simulation data
        datacontainer.add_attributes(path='/MetaData/DomainValues', DomainType="Time in s")
        
        # Iterate through all keys and save the values in the datacontainer ==> 
        # If one key does not exist the variable in the datapoint will stay None
        for key in data.keys():
            match key:
                case 'Tdata':
                    datacontainer.update_dataset(path='/Data/Tdata', data=data[key])
                case 'GroundTruth':
                    # Check if file is old or new format for the label ids
                    if isinstance(data[key], mat73.core.AttrDict):
                        datacontainer.add_attributes(path='/GroundTruth/DefectMask', LabelIds=data[key].LabelIds)
                        datacontainer.update_dataset(path='/GroundTruth/DefectMask', data=data[key].DefectMask)
                    else:
                        # datacontainer.update_attributes(path='GroundTruth/DefectMask', LabelIds=None)
                        datacontainer.update_dataset(path='/GroundTruth/DefectMask', data=data[key])
                case 'Time':
                    datacontainer.update_dataset(path='/MetaData/DomainValues', data=data[key])
                case 'LookUpTable':
                    datacontainer.update_dataset(path='/MetaData/LookUpTable', data=data[key])
                case 'ExcitationSignal':
                    datacontainer.update_dataset(path='/MetaData/ExcitationSignal', data=data[key])
                case 'ComsolParameters':
                    # Convert Comsol Parameters to a json string
                    converted_comsol_parameters = [
                        [item.item() if isinstance(item, np.ndarray) else item for item in sublist]
                        for sublist in data[key]
                    ]

                    # Replace ' with " to make it a valid json string
                    converted_comsol_parameters = str(converted_comsol_parameters).replace("'", '"')

                    # Construct json string and write it 
                    json_string = json.dumps(converted_comsol_parameters, indent=4)
                    datacontainer.add_attributes(path='/MetaData', SimulationParameter=json_string)
                    
                case 'NoiseLevel':
                    datacontainer.add_attributes(path='/Data/Tdata', NoiseLevel=data[key])
                case 'Shapes':
                    # Convert the Shapes into a Python dict first:
                    shapes = {data[key].Names[i] : data[key].Numbers[i] for i in range(len(data[key].Names))}
                    datacontainer.add_attributes(path='/MetaData', Shapes=shapes)

        # Return the constructed datapoint
        return datacontainer