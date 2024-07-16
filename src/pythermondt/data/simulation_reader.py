import mat73
import numpy as np
import json
from typing import Tuple
from .base_reader import BaseReader
from .data_container import DataContainer

class SimulationReader(BaseReader):
    def __init__(self, source:str, file_extension: str | Tuple[str, ...] = '.mat', filter_files: bool = True, cache_paths: bool = True):
        # Call the constructor of the BaseLoader and set the file extension 
        super().__init__(source, file_extension, filter_files=filter_files, cache_paths=cache_paths)
         
    def _read_data(self, file_path:str) -> DataContainer:
        #Load the mat file
        data = mat73.loadmat(file_path, use_attrdict=True)['SimResult']

        # Create a empty Datapoint
        # Domaintype will always be time in case of the Simulation data
        datacontainer = DataContainer()
        datacontainer.update_attributes(group_name='MetaData', dataset_name='DomainValues', DomainType="Time in s")
        
        # Iterate through all keys and save the values in the datapoint ==> 
        # If one key does not exist the variable in the datapoint will stay None
        for key in data.keys():
            match key:
                case 'Tdata':
                    datacontainer.fill_dataset(group_name='Data', dataset_name='Tdata', data=data[key])
                case 'GroundTruth':
                    # Check if file is old or new format
                    if isinstance(data[key], mat73.core.AttrDict):
                        datacontainer.update_attributes(group_name='GroundTruth', dataset_name='DefectMask', LabelIds=data[key].LabelIds)
                        datacontainer.fill_dataset(group_name='GroundTruth', dataset_name='DefectMask', data=data[key].DefectMask)
                    else:
                        datacontainer.update_attributes(group_name='GroundTruth', dataset_name='DefectMask', LabelIds=None)
                        datacontainer.fill_dataset(group_name='GroundTruth', dataset_name='DefectMask', data=data[key])
                case 'Time':
                    datacontainer.fill_dataset(group_name='MetaData', dataset_name='DomainValues', data=data[key])
                case 'LookUpTable':
                    datacontainer.fill_dataset(group_name='MetaData', dataset_name='LookUpTable', data=data[key])
                case 'ExcitationSignal':
                    datacontainer.fill_dataset(group_name='MetaData', dataset_name='ExcitationSignal', data=data[key])
                case 'ComsolParameters':
                    # Convert Comsol Parameters to a json string
                    # Convert ndarrays in scalar                    
                    converted_comsol_parameters = [[item.item() if isinstance(item, np.ndarray) else item for item in sublist] for sublist in data[key]]

                    # Construct json string
                    json_string = json.dumps(converted_comsol_parameters, indent=4)
                    datacontainer.fill_dataset(group_name='MetaData', dataset_name='SimulationParameter', data=json_string)
                    
                case 'NoiseLevel':
                    datacontainer.update_attributes(group_name='Data', dataset_name='Tdata', NoiseLevel=data[key])
                case 'Shapes':
                    # Convert the Shapes into a Python dict first:
                    shapes = {data[key].Names[i] : data[key].Numbers[i] for i in range(len(data[key].Names))}
                    datacontainer.update_attributes(group_name='MetaData', dataset_name='SimulationParameter', Shapes=shapes)

        # Return the constructed datapoint
        return datacontainer