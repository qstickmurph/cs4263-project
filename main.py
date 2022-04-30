from datetime import date, datetime
from tracemalloc import start
from dateutil.relativedelta import relativedelta
from matplotlib.pyplot import fill

import numpy as np
import pandas as pd
import tensorflow as tf

from cs4263_project.data import *

start_date = datetime(year=2004, month=1, day=1)
end_date   = datetime(year=2019, month=6, day=28)

nymex_df = read_nymex(
    file="data/US_EIA_NYMEX.csv",
    start_date=start_date,
    end_date=end_date,
    interpolate=True,
    fill_missing_dates=True)

google_trends_df = read_google_trends(
    file="data/google_trends_dataset.csv",
    keywords=["Natural Gas","Oil","Coal","Nuclear Power","Wind Power",
              "Hydroelectric","Solar Power","Gold","Silver","Platinum","Copper",
              "Biofuel","Recession","CPI"],
    categories=[904,904,904,0,0,0,0,904,904,904,904,0,0,0],
    start_date=start_date,
    end_date=end_date)