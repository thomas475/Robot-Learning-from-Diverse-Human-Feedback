import os
import pandas as pd
import json


def process_csv(file):
    df = pd.read_csv(file)
    
    results = []
    
    for _, row in df.iterrows():
        name_parts = row['name'].split('_')
        
        environment = name_parts[0]
        algorithm = name_parts[1]
        seed = name_parts[-1]
        auxiliary_models = name_parts[2:-1]
        
        result = {
            'environment': environment,
            'algorithm': algorithm,
            'auxiliary_models': auxiliary_models,
            'seed': seed,
            'score': row['d4rl_normalized_score']
        }
        
        results.append(result)
    
    return results
    
def combine_scores(data):
    combined_data = {}
    
    for entry in data:
        key = (entry['environment'], entry['algorithm'], tuple(entry['auxiliary_models']))
        if key not in combined_data:
            combined_data[key] = []
        
        combined_data[key].append(entry['score'])
    
    result = []
    for key, scores in combined_data.items():
        result.append({
            'environment': key[0],
            'algorithm': key[1],
            'auxiliary_models': list(key[2]),
            'scores': scores
        })
    
    return result

def process_all_csv_files():
    all_data = []

    csv_files = [file for file in os.listdir() if file.endswith('.csv')]
    
    for csv_file in csv_files:
        data = process_csv(csv_file)
        all_data.extend(data)
        
    combined_data = combine_scores(all_data)

    with open('scores.json', 'w') as json_file:
        json.dump(combined_data, json_file, indent=4)

    print(f"Processed {len(csv_files)} CSV files and saved the results to scores.json")

process_all_csv_files()
