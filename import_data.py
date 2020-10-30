import pandas as pd
from pandas_profiling import ProfileReport  # Quick preliminary data exploration

import depute_api


api = depute_api.CPCApi()

deputies_json = api.parlementaires()
deputies_df = pd.json_normalize(deputies_json)

profile = ProfileReport(deputies_df, title="Pandas Profiling Report #1")
profile.to_file("df_profile_report.html")
