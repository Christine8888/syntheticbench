import torch
from tqdm import tqdm
from model import Model
import graph
import json
import numpy as np

class EvaluationSuite:
    def __init__(self, model, graph, n_runs=20,
                 max_length=500, do_sample=False, temperature=0.0):
        self.lm = model
        self.graph = graph
        self.n_runs = n_runs
        self.max_length = max_length
        self.do_sample = do_sample
        self.temperature = temperature

    def run_evaluation(self):
        results = []
        
        for _ in tqdm(range(self.n_runs)):
            # generate starter path
            starter_path = None
            while starter_path is None:
                try:
                    starter_path = self.graph.generate_random_path(np.random.choice(self.graph.nodes), 20)
                except ValueError:
                    continue
            
            # generate prompt
            matrix = self.graph.print_adjacency_matrix()
            prompt = matrix + '\nSTARTER PATH:' + self.graph.print_path(starter_path)

            # generate and parse text
            generated_text = self.lm.generate_text(prompt)
            predicted_path_str = generated_text.split('STARTER PATH:')[1].strip()
            predicted_path = self.graph.str_to_path(predicted_path_str)
            if len(predicted_path) < 2:
                print('parse failed')
                continue
            
            results.append({
                'starter_path': starter_path,
                'predicted_path': predicted_path,
                'prompt': prompt,
                'output': generated_text
            })
        return results

def survival_curve(starter_path, predicted_path, graph):
    valid_after_starter = 0
    for i in range(len(starter_path), len(predicted_path) - 1):
        if graph.is_valid_path(predicted_path[:i+2]):
            valid_after_starter += 1
        else:
            break
    return valid_after_starter

def adjacency_usage(starter_path, predicted_path, graph):
    starter_edges = set(tuple(starter_path[i:i+2]) for i in range(len(starter_path) - 1))
    new_correct_edges = 0
    for i in range(len(predicted_path) - 1):
        edge = tuple(predicted_path[i:i+2])
        if edge not in starter_edges and graph.is_valid_path(predicted_path[i:i+2]):
            new_correct_edges += 1
    return new_correct_edges

if __name__ == '__main__':
    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    print('loading model ' + MODEL)
    test_model = Model(MODEL)
    print('loading evaluation suite...')

    test_graph = graph.BinaryERGraph(10, 0.2)
    suite = EvaluationSuite(test_model, test_graph, n_runs=10)
    results = suite.run_evaluation()
    # write full result dictionary to file
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    sc_values = []
    au_values = []

    for r in results:
        sc_values.append(survival_curve(r['starter_path'], r['predicted_path'], test_graph))
        au_values.append(adjacency_usage(r['starter_path'], r['predicted_path'], test_graph))

    print(sc_values)
    print(au_values)

    print('average survival:', sum(sc_values)/len(sc_values))
    print('average adjacency usage:', sum(au_values)/len(au_values))
