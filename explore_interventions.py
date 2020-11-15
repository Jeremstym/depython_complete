# import libraries
import pandas as pd

# import API module
import depute_api

# load deputies dataframe
api = depute_api.CPCApi()
deputies_json = api.parlementaires()
deputies_df = pd.json_normalize(deputies_json)

# get list of all *groupes parlementaires*
groupes = deputies_df["groupe_sigle"].unique()

# Intermediary functions
def deputies_of_group(group, n_deputies):
    all_names = deputies_df[deputies_df["groupe_sigle"] == group]["nom"]
    return all_names[:n_deputies]

def interventions_of_group(group, n_deputies = 15, n_sessions = 10):
    names = deputies_of_group(group, n_deputies)
    print(names)
    interventions = []
    for name in names:
        print(name)
        interventions += [[group, name, api.interventions(name, n_sessions)]]
    return interventions


# Populate list of interventions (this step takes some time)
interventions_from_all_groups = []

for groupe in groupes[2:]:
    interventions_from_all_groups += interventions_of_group(groupe)


# Convert to `DataFrame` and export to csv
interventions_df = pd.DataFrame(
    interventions_from_all_groups,
    columns = ["groupe", "nom", "interventions"]
)

path = "/path/to/data_folder"
interventions_df.to_csv(path + "/interventions.csv")
