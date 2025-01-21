from pythermondt.data import DataContainer

def test_serialize_deserialize(filled_container: DataContainer):
    """Test serialization and deserialization of DataContainer."""
    # Serialize the DataContainer
    hdf5_bytes = filled_container.serialize_to_hdf5()

    # Deserialize the DataContainer
    deserialized_container = DataContainer(hdf5_bytes)

    # Check if the deserialized container is equal to the original container
    assert deserialized_container == filled_container