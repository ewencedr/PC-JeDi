import pandas as pd
import numpy as np
from pathlib import Path

import yaml

ABS_PATH = Path("/srv/beegfs/scratch/users/s/senguptd/jet_diffusion/")

#epic jedi 30
epic_jedi_30_cond = ABS_PATH / "epic_jedi_30/2023-05-19_10-03-09-602489/"
epic_jedi_30_uncond = ABS_PATH / "epic_uncond_30/2023-05-19_09-45-57-057758/"

#epic jedi 150
epic_jedi_150_cond = ABS_PATH / "epic_jedi_150/2023-05-17_13-28-26-652019/"
epic_jedi_150_uncond = ABS_PATH / "epic_uncond_150/2023-05-15_16-33-05-126098/"

#PC jedi 150
pc_jedi_150_cond = ABS_PATH / "pc_jedi_150/2023-05-19_11-01-28-016315/"
pc_jedi_150_uncond = ABS_PATH / "pc_uncond_150/2023-05-19_10-45-33-374505/"

def format_float_with_precision(series, precision=4):
    return series.apply(lambda x: f"{x:.{precision}f}")

def format_float_with_scientific(series, precision=3):
    return series.apply(lambda x: f"{x:.{precision}e}")


def get_scores(path, sampler_list=["ddim_200","em_200"]):

    scores = {}
    for sampler in sampler_list:
        # Read the yaml file
        yaml_file = path / f"outputs/yaml_scores/{sampler}"
        with open(yaml_file, 'r') as stream:
            try:
                scores[sampler] = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    return scores

def shape_df(df):
    df = pd.DataFrame(df)
    mapper = {
        'cov': format_float_with_precision,
        'fpnd': format_float_with_precision,
        'm_mae': format_float_with_precision,
        'mmd': format_float_with_precision,
        'pt_mae': format_float_with_precision,
        'w1efp': format_float_with_scientific,
        'w1efp_err': format_float_with_scientific,
        'w1m': format_float_with_scientific,
        'w1m_err': format_float_with_scientific,
        'w1p': format_float_with_scientific,
        'w1p_err': format_float_with_scientific,
    }

    transposed_df = df.transpose()
    transposed_df = transposed_df.apply(lambda x: mapper[x.name](x))
    transposed_df["fpnd"] = "-" if (transposed_df["fpnd"] == "-999.0000").any() else transposed_df["fpnd"]

    transposed_df["W1EFP"] = transposed_df["w1efp"].astype(str) + " +- " + transposed_df["w1efp_err"].astype(str)
    transposed_df["W1M"] = transposed_df["w1m"].astype(str)+ " +- " + transposed_df["w1m_err"].astype(str)
    transposed_df["W1P"] = transposed_df["w1p"].astype(str) + " +- " + transposed_df["w1p_err"].astype(str)

    transposed_df = transposed_df.drop(columns=["w1efp","w1efp_err","w1m","w1m_err","w1p","w1p_err"])
    return transposed_df


epic_jedi_150_cond_scores = get_scores(epic_jedi_150_cond, sampler_list=["ddim_200","em_200"])
epic_jedi_150_uncond_scores = get_scores(epic_jedi_150_uncond, sampler_list=["ddim_200","em_200"])
pc_jedi_150_cond_scores = get_scores(pc_jedi_150_cond, sampler_list=["ddim_200","em_200"])
pc_jedi_150_uncond_scores = get_scores(pc_jedi_150_uncond, sampler_list=["ddim_200","em_200"])

epic_jedi_30_cond_scores = get_scores(epic_jedi_30_cond, sampler_list=["ddim_200","em_200"])
epic_jedi_30_uncond_scores = get_scores(epic_jedi_30_uncond, sampler_list=["ddim_200","em_200"])

epic_jedi_150_cond_scores = shape_df(epic_jedi_150_cond_scores)
epic_jedi_150_uncond_scores = shape_df(epic_jedi_150_uncond_scores)
pc_jedi_150_cond_scores = shape_df(pc_jedi_150_cond_scores)
pc_jedi_150_uncond_scores = shape_df(pc_jedi_150_uncond_scores)

epic_jedi_30_cond_scores = shape_df(epic_jedi_30_cond_scores)
epic_jedi_30_uncond_scores = shape_df(epic_jedi_30_uncond_scores)

# print(epic_jedi_150_cond_scores)
                
# Collect all the 30 constituent models --> categorise them between conditional and unconditional models and make one df.

model_30_scores = pd.concat([epic_jedi_30_cond_scores,epic_jedi_30_uncond_scores],axis=0, keys=["Conditional", "Unconditional"])

epic_model_150_scores = pd.concat([epic_jedi_150_cond_scores,epic_jedi_150_uncond_scores], axis=0, keys=["Conditional", "Unconditional"])
pc_model_150_scores = pd.concat([pc_jedi_150_cond_scores,pc_jedi_150_uncond_scores], axis=0, keys=["Conditional", "Unconditional"])

model_150_scores = pd.concat([epic_model_150_scores,pc_model_150_scores], axis=0, keys=["EPIC", "PC"])

print(model_150_scores.style.to_latex())

print("===================")
print(model_30_scores.style.to_latex())

print(model_30_scores)