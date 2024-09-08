import os
import json


def create_language_extension_mapping(base_path):
    language_extension_map = {}

    # Traverse all directories under the base path
    for folder in os.listdir(base_path):
        if folder.startswith('source_code_'):

            folder_path = os.path.join(base_path, folder)
            # print(folder_path)

            # Read the JSON file
            json_file = os.path.join(folder_path, f'{folder}.json')
            with open(json_file, 'r') as f:
                metadata = json.load(f)

            programming_language = metadata['programming_language']
            print(programming_language)

            for file in os.listdir(folder_path):
                if not file.endswith('.json'):
                    language_extension_map[programming_language] = file.split(
                        '.')[-1]
                    break

    return language_extension_map


# Usage
base_path = os.path.join(os.path.dirname(
    __file__), os.pardir, 'data', 'dataset-source-codes')
language_extension_map = create_language_extension_mapping(base_path)

# Print the resulting mapping
print(json.dumps(language_extension_map, indent=2))

# Optionally, save the mapping to a JSON file
with open(os.path.join(os.path.dirname(__file__), "language_extension_map.json"), 'w') as f:
    json.dump(language_extension_map, f, indent=2)

print("Mapping saved to language_extension_map.json")
