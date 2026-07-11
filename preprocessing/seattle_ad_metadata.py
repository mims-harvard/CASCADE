"""
Seattle-AD (ACT) donor/clinical metadata preprocessing helpers: age binning,
cause-of-death normalisation, and ATC medication-code grouping (Methods 9.3).
"""
import numpy as np
import pandas as pd

cause_of_death_mapping = {
    12: "suicide, hanging",
    4: "drowning",
    10: "sudden unexpected death, likely due to epileptic seizure",
    3: "dilated cardiomegaly",
    11: "sudden unexpected death, likely due to epileptic seizure or cardiac arrhythmia",
    5: "multiorgan failure",
    0: "aortic stenosis",
    8: "smoke inhalation",
    2: "cardiac arrhythmia",
    7: "opioid intoxication",
    9: "status asthmaticus",
    1: "car accident",
    6: "multiple injuries",
}

act_codes = [
    'A08AA51', 'A11HA02', 'C02LC01', 'C02LC51', 'N02AA02', 'N02CX02',
    'N03AF01', 'N03AG01', 'N03AX14', 'N05AH04', 'N05CH01', 'N06AB03',
    'N06AB06', 'N06AB08', 'N06CA03', 'S01EA04', 'A08AA', 'A11HA', 'C02LC',
    'N02AA', 'N02CX', 'N03AF', 'N03AG', 'N03AX', 'N05AH', 'N05CH', 'N06AB',
    'N06CA', 'S01EA', 'A08A', 'A11H', 'C02L', 'N02A', 'N02C', 'N03A', 'N05A',
    'N05C', 'N06A', 'N06C', 'S01E', 'A08', 'A11', 'C02', 'N02', 'N03', 'N05',
    'N06', 'S01', 'A', 'C', 'N', 'S',
]

replacement_age = {
    "below-50": ['29-year-old human stage', '42-year-old human stage',
                 '43-year-old human stage', '50-year-old human stage'],
    "60-70": ['60-year-old human stage', '65-year-old human stage',
              '68-year-old human stage', '69-year-old human stage',
              '70-year-old human stage'],
    "70-80": ['72-year-old human stage',
              '75-year-old human stage', '77-year-old human stage',
              '78-year-old human stage', '80 year-old and over human stage'],
    "80-90": ['80-year-old human stage', '81-year-old human stage',
              '82-year-old human stage', '83-year-old human stage',
              '84-year-old human stage', '85-year-old human stage',
              '86-year-old human stage', '87-year-old human stage',
              '88-year-old human stage', '89-year-old human stage'],
}

medications_mapping_dict = {
    2: ["Quetiapine", "Fluvoxamine"],
    4: ["Valproic acid", "Levetiracetam", "Vitamin B6", "Fluoxetine"],
    3: ["Topiramate"],
    1: ["Opioids"],
    0: ["Clonidine", "Sertraline", "Carbamazepine", "Melatonin"],
}

# ATC drug-code hierarchy (levels 1-5) for the medications used to derive `medications_mapping_dict`
mapping_data = {
    "Medications_mapping": [
        "Topiramate", "Clonidine", "Sertraline", "Carbamazepine", "Melatonin",
        "Opioids", "Valproic acid", "Levetiracetam", "Vitamin B6", "Fluoxetine",
        "Quetiapine", "Fluvoxamine",
    ],
    "Level 5": [
        ["A08AA51"],
        ["S01EA04", "C02LC01", "N02CX02", "C02LC51"],
        ["N06AB06"],
        ["N03AF01"],
        ["N05CH01"],
        ["N02AA02"],
        ["N03AG01"],
        ["N03AX14"],
        ["A11HA02"],
        ["N06AB03", "N06CA03"],
        ["N05AH04"],
        ["N06AB08"],
    ],
    "Level 4": [
        ["A08AA"],
        ["S01EA", "C02LC", "N02CX", "C02LC"],
        ["N06AB"],
        ["N03AF"],
        ["N05CH"],
        ["N02AA"],
        ["N03AG"],
        ["N03AX"],
        ["A11HA"],
        ["N06AB", "N06CA"],
        ["N05AH"],
        ["N06AB"],
    ],
    "Level 3": [
        ["A08A"],
        ["S01E", "C02L", "N02C", "C02L"],
        ["N06A"],
        ["N03A"],
        ["N05C"],
        ["N02A"],
        ["N03A"],
        ["N03A"],
        ["A11H"],
        ["N06A", "N06C"],
        ["N05A"],
        ["N06A"],
    ],
    "Level 2": [
        ["A08"],
        ["S01", "C02", "N02", "C02"],
        ["N06"],
        ["N03"],
        ["N05"],
        ["N02"],
        ["N03"],
        ["N03"],
        ["A11"],
        ["N06", "N06"],
        ["N05"],
        ["N06"],
    ],
    "Level 1": [
        ["A"],
        ["S", "C", "N", "C"],
        ["N"],
        ["N"],
        ["N"],
        ["N"],
        ["N"],
        ["N"],
        ["A"],
        ["N", "N"],
        ["N"],
        ["N"],
    ],
}


def bin_age(dict_replacement, output_dict):
    inverted_dict = {}
    for key, values in dict_replacement.items():
        for value in values:
            inverted_dict[value] = key

    age_binned = pd.Series(output_dict["development_stage"]).replace(inverted_dict)
    return age_binned.tolist()


def generate_binary_columns(row, conditions, column='Other diagnoses'):
    if pd.isna(row[column]):
        return [np.nan] * len(conditions)
    condition_name = str(row[column])
    return [1 if str(condition).lower() in condition_name.lower() else 0 for condition in conditions]
