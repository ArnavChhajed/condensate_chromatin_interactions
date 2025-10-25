import h5py

path = r"C:\Users\arnav\Personal - Arnav Chhajed\Northwestern\genome_organization\examples\arnav\test_output\blocks_10-10.h5"

with h5py.File(path, "r") as f:
    print("Top-level keys:", list(f.keys()))
    
    # Each group corresponds to a frame
    first_frame = list(f.keys())[0]
    print("Keys inside first frame:", list(f[first_frame].keys()))
    
    # Position shape
    pos = f[first_frame]["pos"][:]
    print("Shape of pos:", pos.shape)
    print("First few positions:\n", pos[:5])
