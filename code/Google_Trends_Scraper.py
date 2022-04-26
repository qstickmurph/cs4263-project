# %%
import pandas as pd
import pytrends
from pytrends.request import TrendReq
import glob
import os
pytrend = TrendReq()

# %%
pytrend.categories()

# %%
KEYWORDS            = ["Natural Gas","Oil","Coal","Nuclear Power","Wind Power","Hydroelectric","Solar Power","Gold","Silver","Platinum","Copper","Biofuel","Recession","CPI"]
KEYWORDS_CATEGORIES = [904,          904,  904,   0,               0,          0,               0,           904,   904,     904,       904,     0,        0,          0]
KEYWORDS_CODES=[pytrend.suggestions(keyword=i)[0] for i in KEYWORDS] 
df_CODES= pd.DataFrame(KEYWORDS_CODES)
df_CODES

# %%
EXACT_KEYWORDS=df_CODES['mid'].to_list() #
DATE_INTERVAL='2013-01-01 2019-6-30' # Jan 2013 - June 2019
COUNTRY=["US"] # ISO country code
CATEGORY=0 # Use this link to select categories
SEARCH_TYPE='' #default is 'web searches',others include 'images','news','youtube','froogle' (google shopping)

# %%
Individual_EXACT_KEYWORD = list(zip(*[iter(EXACT_KEYWORDS)]*1, KEYWORDS_CATEGORIES))
Individual_EXACT_KEYWORD = [list(x) for x in Individual_EXACT_KEYWORD]
dicti = {}
i = 1
for Country in COUNTRY:
    for keyword, category in Individual_EXACT_KEYWORD:
        pytrend.build_payload(kw_list=[keyword], 
                              timeframe = DATE_INTERVAL, 
                              geo = Country, 
                              cat=category,
                              gprop=SEARCH_TYPE) 
        dicti[i] = pytrend.interest_over_time()
        i+=1
df_trends = pd.concat(dicti, axis=1)

# %%
df_trends.columns = df_trends.columns.droplevel(0) #drop outside header
df_trends = df_trends.drop('isPartial', axis = 1) #drop "isPartial"
df_trends.reset_index(level=0,inplace=True) #reset_index
df_trends.columns=["date","Natural Gas","Oil","Coal","Nuclear Power","Wind Power","Hydroelectric","Solar Power","Gold","Silver","Platinum","Copper","Biofuel","Recession","CPI"] #change column names

# %%
print(df_trends)