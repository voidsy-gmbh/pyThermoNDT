import mat73
import numpy as np
import json
from typing import Tuple
from ._base_reader import _BaseReader
from ..data import DataContainer

class SimulationReader(_BaseReader):
    def __init__(self, source:str, file_extension: str | Tuple[str, ...] = '.mat', cache_paths: bool = True):
        # Call the constructor of the BaseLoader and set the file extension 
        super().__init__(source, file_extension, cache_paths=cache_paths)
         
    def _read_data(self, file_path:str) -> DataContainer:
        #Load the mat file
        data = mat73.loadmat(file_path, use_attrdict=True)['SimResult']

        # Create an empty DataContainer
        datacontainer = DataContainer()

        # Add source as an attribute
        datacontainer.update_attributes(path='MetaData', Source='Simulation')

        # Domaintype will always be time in case of the Simulation data
        datacontainer.update_attributes(path='MetaData/DomainValues', DomainType="Time in s")
        
        # Iterate through all keys and save the values in the datapoint ==> 
        # If one key does not exist the variable in the datapoint will stay None
        for key in data.keys():
            match key:
                case 'Tdata':
                    datacontainer.fill_dataset(path='Data/Tdata', data=data[key])
                case 'GroundTruth':
                    # Check if file is old or new format for the label ids
                    if isinstance(data[key], mat73.core.AttrDict):
                        datacontainer.update_attributes(path='GroundTruth/DefectMask', LabelIds=data[key].LabelIds)
                        datacontainer.fill_dataset(path='GroundTruth/DefectMask', data=data[key].DefectMask)
                    else:
                        datacontainer.update_attributes(path='GroundTruth/DefectMask', LabelIds=None)
                        datacontainer.fill_dataset(path='GroundTruth/DefectMask', data=data[key])
                case 'Time':
                    datacontainer.fill_dataset(path='MetaData/DomainValues', data=data[key])
                case 'LookUpTable':
                    datacontainer.fill_dataset(path='MetaData/LookUpTable', data=data[key])
                case 'ExcitationSignal':
                    datacontainer.fill_dataset(path='MetaData/ExcitationSignal', data=data[key])
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
                    datacontainer.update_attributes(path='MetaData', SimulationParameter=json_string)
                    
                case 'NoiseLevel':
                    datacontainer.update_attributes(path='Data/Tdata', NoiseLevel=data[key])
                case 'Shapes':
                    # Convert the Shapes into a Python dict first:
                    shapes = {data[key].Names[i] : data[key].Numbers[i] for i in range(len(data[key].Names))}
                    datacontainer.update_attributes(path='MetaData', Shapes=shapes)

        # Return the constructed datapoint
        return datacontainer