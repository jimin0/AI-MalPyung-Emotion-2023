import json
import numpy as np

def jsonlload(fname):
    with open(fname, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")
        j_list = [json.loads(line) for line in lines]
    return j_list

def ensemble_from_files(file_paths, primary_threshold=0.5, secondary_threshold=0.45):
    # Load logits and sentences from each file
    data_sets = [jsonlload(file_path) for file_path in file_paths]

    # Initialize a list to store the ensemble results
    ensemble_results = []

    # Go through each data point and ensemble
    for idx in range(len(data_sets[0])):  # Assuming all files have the same number of lines
        # Collect logits for the current data point across all files
        logits_list = [np.array(data_set[idx]["logits"]) for data_set in data_sets]

        # Average the logits
        avg_logits = np.mean(logits_list, axis=0)

        # Make final decision based on the primary threshold
        final_decision = {}
        emotions = ["joy", "anticipation", "trust", "surprise", "disgust", "fear", "anger", "sadness"]
        for i, emotion in enumerate(emotions):
            final_decision[emotion] = "True" if avg_logits[i] > primary_threshold else "False"

        # If all emotions are False, set the emotion with the highest logit to True
        # and any emotion that exceeds the secondary threshold
        if all(value == "False" for value in final_decision.values()):
            highest_logit_index = np.argmax(avg_logits)
            final_decision[emotions[highest_logit_index]] = "True"

            # # Additionally, set any emotion that exceeds the secondary threshold to True
            # for i, logit in enumerate(avg_logits):
            #     if logit > secondary_threshold:
            #         final_decision[emotions[i]] = "True"

        ensemble_results.append(final_decision)

    return ensemble_results



def save_ensemble_results_to_jsonl_corrected(data_path, ensemble_results, output_path):
    """
    Save the ensemble results to a jsonl file without disrupting the original format.
    
    Parameters:
    - data_path: Path to the original jsonl file (to get the structure)
    - ensemble_results: List of dictionaries containing the ensemble results
    - output_path: Path to save the ensemble results
    """
    with open(data_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f.readlines()]
        
    # Update the output field with the ensemble results
    for idx, entry in enumerate(entries):
        entry["output"] = ensemble_results[idx]
        entries[idx] = entry
    
    # Save the updated entries to the output file
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            

if __name__ == "__main__":

    file_paths = [
        '../Ensemble/매멋logits.jsonl',
        '../Ensemble/천둥오리logits.jsonl',
        '../Ensemble/코끼리logits.jsonl',
        '../Ensemble/은색오리logits5300logits.jsonl',
        '../Ensemble/은색오리logits5400logits.jsonl',
        '../Ensemble/황색오리logits4200logits.jsonl',
        '../Ensemble/황색오리logits4300logits.jsonl',
        '../Ensemble/동색오리logits5000logits.jsonl',
        '../Ensemble/동색오리logits4500logits.jsonl',
        '../Ensemble/프렌치불독logits.jsonl',
        '../Ensemble/시바개logits.jsonl',
    ]
    # Call the ensemble function (for demonstration, using the previously loaded files)
    ensemble_results_demo = ensemble_from_files(file_paths)
    # Save the ensemble results to the original file (for demonstration, using the previously loaded file)
    #save_ensemble_results_to_jsonl_corrected('./results/Kfold합본/산양logits.jsonl', ensemble_results_demo, './results/Kfold합본/동물분류27.jsonl')
    save_ensemble_results_to_jsonl_corrected('/home/nlplab/hdd2/대회제출/Ensemble/가젤왕.jsonl', ensemble_results_demo, '../Ensemble/최종제출.jsonl')

