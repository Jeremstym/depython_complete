import pandas as pd

import depute_api


api = depute_api.CPCApi()

deputies_json = api.parlementaires()
deputies_df = pd.json_normalize(deputies_json)


deputies_df.head()
