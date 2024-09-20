import json
import itertools
import numpy as np

def abbreviate_auxiliary_models(aux_models):
    if len(aux_models) == 0:
        return "BASE"
    abbreviations = [''.join([char for char in model if char.isupper()][:2]) for model in aux_models]
    return "+".join(abbreviations)

def generate_latex_table(data, algorithm, environment_order, model_order, calculate_average, show_deviation):
    data = [entry for entry in data if entry['algorithm'] == algorithm]

    latex = "\\begin{table}[t]\n\\centering\n\\scriptsize\n\\renewcommand{\\arraystretch}{1.2}\n"
    latex += "\\begin{tabular}{|r|" + ">{\centering\\arraybackslash}p{1.9cm}|" * len(environment_order) + "}\n"
    latex += "\\hline\n"

    latex += "\\multirow{2}{*}{\\textbf{method}} & \\multicolumn{3}{c|}{\\textbf{dataset}} \\\\\n"
    latex += "\\cline{2-4}\n"
    latex += "& " + " & ".join(environment_order) + " \\\\\n"
    latex += "\\hline\n"
    
    scores_dict = {env: [] for env in environment_order}

    for aux_comb in model_order:
        latex += f"{aux_comb} "
        for env in environment_order:
            score = "$-$"
            for entry in data:
                if (entry['environment'] == env and abbreviate_auxiliary_models(entry['auxiliary_models']) == aux_comb and not any(np.isnan(entry['scores']))):
                    mean_score = np.mean(entry['scores'])
                    if show_deviation:
                        deviation_score = np.std(entry['scores'])
                        score = (mean_score, deviation_score)
                    else:
                        score = mean_score
            if show_deviation and isinstance(score, tuple):
                latex += f"& {score[0]:.2f} $\\pm$ {score[1]:.2f} "
            else:
                latex += f"& {score:.2f} " if isinstance(score, float) else "& $-$ "
                
            if isinstance(score, tuple):
                scores_dict[env].append(score[0])
            elif isinstance(score, float):
                scores_dict[env].append(score)
                
        latex += "\\\\\n"
    
    if calculate_average:
        latex += "\\hline\n\\textbf{average} "
        for env in environment_order:
            if scores_dict[env]:
                avg_score = np.mean(scores_dict[env])
                if show_deviation:
                    avg_deviation = np.std(scores_dict[env])
                    latex += f"& {avg_score:.2f} $\\pm$ {avg_deviation:.2f} "
                else:
                    latex += f"& {avg_score:.2f} "
            else:
                latex += "& $-$ "
        latex += "\\\\\n"

    latex += "\\hline\n\\end{tabular}\n\\vspace{0.1cm}\n"
    latex += "\\caption{Mean scores and deviations for the " + algorithm + " algorithm with selected auxiliary models and combinations.}\n"
    latex += "\\label{tab:" + algorithm.lower() + "Results}\n"
    latex += "\\end{table}"

    return latex

def main():
    with open('scores.json', 'r') as f:
        data = json.load(f)

    algorithms = ["IQL", "DiffusionQL"]
    environment_order = ["kitchen-complete-v0", "kitchen-partial-v0", "kitchen-mixed-v0"]
    model_order = ["BASE", "HC", "HE", "HK", "HC+HE", "HC+HK", "HE+HK", "HC+HE+HK", "RC", "RE"]
    show_deviation = True
    calculate_average = True

    for algorithm in algorithms:    
        latex_table = generate_latex_table(data, algorithm, environment_order, model_order, calculate_average, show_deviation)
        
        file_name = algorithm.lower() + "_scores_table.tex"
        with open(file_name, 'w') as f:
            f.write(latex_table)
        
        print("LaTeX table has been generated and saved to " + file_name)

main()

