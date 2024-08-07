import os
import pickle

def unpickle_and_inspect(file_path, num_samples=5):
    """
    Unpickle a file and print its structure and example content.

    :param file_path: str, path to the pickled file
    :param num_samples: int, number of sample entries to print from the unpickled data
    """
    try:
        # Open the file in binary read mode and unpickle the content
        with open(file_path, 'rb') as file:
            data = pickle.load(file)

        # Print the type of the data
        print(f"Data type: {type(data)}")
        print(f"Data shape: {data.shape}")

        # Print the structure of the data
        if isinstance(data, dict):
            print(f"Keys: {list(data.keys())[:num_samples]}")
            print("Sample Content:")
            for key, value in list(data.items())[:num_samples]:
                print(f"Key: {key} - Type of Value: {type(value)} - Example Value: {value}\n")
        elif isinstance(data, list):
            print(f"Number of items: {len(data)}")
            print("Sample Content:")
            for i, item in enumerate(data[:num_samples]):
                print(f"Index: {i} - Type of Item: {type(item)} - Example Item: {item}\n")
        else:
            # For other data types, simply print the data
            print(f"Example Content: {data}")

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except pickle.UnpicklingError:
        print(f"Error unpickling file: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
directory_path = 'crowdsource_human_labels/antmaze-large-diverse-v2_human_labels/'
for file in os.listdir(directory_path):
    unpickle_and_inspect(os.path.join(directory_path, file))