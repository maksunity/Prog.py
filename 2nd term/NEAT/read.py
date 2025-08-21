import pickle

# Replace 'path_to_your_file.pkl' with the actual path to your PKL file
file_path = r'C:\Users\maksunity\PycharmProjects\Prog_Den\Prog.py\2nd term\NEAT\neat_output_custom\winner_genome_custom.pkl'

# Open the file in binary mode and load the data
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Now 'data' contains the deserialized Python object
print(data)