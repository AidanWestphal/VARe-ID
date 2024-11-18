import ctypes
import os


def test_load_zbar_library():
    # Use the path to the dynamic library you confirmed exists
    lib_path = '/opt/homebrew/opt/zbar/lib/libzbar.dylib'  # or libzbar.0.dylib

    try:
        # Load the library directly from the known path
        zbar_lib = ctypes.cdll.LoadLibrary(lib_path)
        print(f"Successfully loaded ZBar library from: {lib_path}")
    except Exception as e:
        print(f"Failed to load ZBar library. Error: {e}")


if __name__ == "__main__":
    # Optionally add the library path to the environment variable for testing
    os.environ['DYLD_LIBRARY_PATH'] = '/opt/homebrew/opt/zbar/lib'

    test_load_zbar_library()
