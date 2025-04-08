from pythermondt.readers import LocalReader

reader = LocalReader("./tests/assets/simulation_data/s*.mat")

for i, data in enumerate(reader, start=1):
    data.save_to_hdf5(f"./tests/assets/simulation_data/expected{i}.hdf5")
