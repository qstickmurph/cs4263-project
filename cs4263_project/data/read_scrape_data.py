__all__ = [
    "read_nymex",
    "read_google_trends",
]

from datetime import date, datetime
from tracemalloc import start
from dateutil.relativedelta import relativedelta
import math
import time

from numpy import float32, nan, real_if_close
import pandas as pd
from pytrends.request import TrendReq

pytrend = TrendReq()

def read_nymex(
        file, 
        start_date=None, 
        end_date=None, 
        interpolate=True, 
        fill_missing_dates=True) -> pd.DataFrame:

    print(f"Getting nymex data from {start_date.date()} to {end_date.date()}")
    # Read dataset
    nymex_df = pd.read_csv(file, parse_dates=["Date"])
    nymex_df.set_index("Date", inplace=True)

    # Restrict to timeframe
    if start_date is not None and end_date is not None:
        nymex_df = nymex_df.loc[
            nymex_df.index.intersection(
                pd.date_range(start_date, end_date, freq='d'))
        ]

    # Add missing dates
    if fill_missing_dates:
        nymex_df = nymex_df.reindex(
            pd.date_range(nymex_df.index[0], nymex_df.index[-1], freq='d'))   
        nymex_df.index.names = ["Date"]

    # Interpolate NaNs
    if interpolate:
        nymex_df.interpolate(method='time', inplace=True)

    return nymex_df


def read_google_trends(
        file, 
        keywords=[], 
        categories=[], 
        start_date=datetime(year=2004, month=1, day=1), 
        end_date=datetime.today()-relativedelta(days=7)) -> pd.DataFrame:
    
    assert len(keywords) == len(categories), \
        "keywords and categories must have the same length"

    assert start_date >= datetime(year=2004, month=1, day=1), \
        "start_date must be after Jan 1, 2004"

    assert end_date < datetime.today(), \
        "end date must be before today"

    print(f"Getting google trends data for {keywords} from {start_date.date()}"
        + f" to {end_date.date()}")

    def find_breakpoints(datetime_index) -> list:
        if len(datetime_index) == 0:
            return []
        date_bp = []
        first_date = datetime_index[0].date()
        last_date = datetime_index[-1].date()
        curr_date = first_date
        i = 0
        while curr_date < last_date:
            if curr_date == datetime_index[i].date():
                curr_date = curr_date + relativedelta(days=1)
            else:
                date_bp.append([first_date, datetime_index[i-1].date()])
                first_date = datetime_index[i].date()
                curr_date = datetime_index[i+1].date()
            i+=1

        date_bp.append([first_date, last_date])
        return date_bp

    # Read dataset
    try:
        google_trends_df = pd.read_csv(file, 
                                       parse_dates=["Date"])
        google_trends_df.set_index("Date", inplace=True)

    except FileNotFoundError: # file doesn't exist
        google_trends_df = pd.DataFrame(
            index=pd.date_range(start_date, end_date, freq='d'),
            columns=keywords, 
            dtype=float32)

    google_trends_df = google_trends_df.astype(float32)

    
    # Add new keyword columns
    new_keywords = [
        x for x in keywords if x not in google_trends_df.columns
    ]
    google_trends_df[new_keywords] = nan

    # Loop over those breakpoints and add them to our dataset
    for keyword in keywords:
        # Find breakpoints for dates not in our dataset yet
        date_bps = find_breakpoints(
            pd.date_range(start_date, end_date, freq='d').difference(
                    google_trends_df.index
                ).union(
                    google_trends_df[pd.isnull(google_trends_df[keyword])].index
                ))

        for d in range(len(date_bps)):
            start_bp = date_bps[d][0]
            end_bp = date_bps[d][1]
            # Add new dates to index, fill with NaN
            google_trends_df = google_trends_df.reindex(
                google_trends_df.index.union(
                    pd.date_range(start_bp, end_bp, freq='d')),
                fill_value=nan)
            google_trends_df.index.names = ["Date"]

            print(f"Scraping for {keyword} from {start_bp} to {end_bp}")
            google_trends_df.loc[
                pd.date_range(start_bp, end_bp, freq='d'), 
                keyword] = _scrape_google_trends(
                                        [keyword], 
                                        categories,
                                        start_bp,
                                        end_bp,
                                        )[keyword]

        google_trends_df.to_csv(file)

    return google_trends_df.loc[pd.date_range(start_date, end_date, freq='d')]


def _scrape_google_trends(
        keywords, 
        categories, 
        start_date: date,
        end_date: date) -> pd.DataFrame:

    def diff_month(d1, d2):
        return (d1.year - d2.year) * 12 + d1.month - d2.month
    
    # Get pytrend suggestions and store them in exact_keywords
    keywords_codes = [pytrend.suggestions(keyword=i)[0] for i in keywords] 
    df_CODES= pd.DataFrame(keywords_codes)
    exact_keywords = df_CODES['mid'].to_list()

    # Store keywords alongside their respective categories
    individual_exact_keyword = list(zip(*[iter(exact_keywords)]*1, categories))
    individual_exact_keyword = [list(x) for x in individual_exact_keyword]

    # Set default vars
    overall_timeframe = date(year=2004, month=1, day=1).isoformat() \
        + " " \
        + (date.today() - relativedelta(days=1)).isoformat()
    country = "US"
    search_type = ''

    # Split timeframe into 1 month chunks
    dates = []
    temp_date = start_date
    while temp_date <= end_date:
        dates.append(temp_date)
        temp_date = temp_date + relativedelta(months=6)

    # Find overall timeframe

    word_id = 0
    trend_dict = {}
    for keyword, category in individual_exact_keyword:
        trend_dict[keyword] = pd.DataFrame()
        # Fetch overall google trends for normalizing
        pytrend.build_payload(kw_list=[keyword], 
                                timeframe=overall_timeframe,
                                geo=country, 
                                cat=category,
                                gprop=search_type)
        overall_data = pytrend.interest_over_time()
        
        time.sleep(5)
        month_num = diff_month(start_date, datetime(year=2004,month=1,day=1))
        for i in range(len(dates)):
            pytrend.build_payload(
                            kw_list=[keyword], 
                            timeframe = dates[i].isoformat() 
                                        + " " 
                                        + (dates[i] 
                                            + relativedelta(months=6)
                                            - relativedelta(days=1)).isoformat(), 
                            geo = country, 
                            cat=category,
                            gprop=search_type)
            time.sleep(5) # sleep to prevent google shutting us down
            month_data = pytrend.interest_over_time()
            month_data = month_data.reindex(
                pd.date_range(
                    dates[i],
                    dates[i]+relativedelta(months=6)-relativedelta(days=1),
                    freq='d'))
            month_data.index.names = ["Date"]

            #print(
            #    pd.date_range(
            #        dates[i],
            #        dates[i]+relativedelta(months=6)-relativedelta(days=1),
            #        freq='d').difference(month_data.index))

            # normalize data based on overall_data
            for month in range(6):
                if (dates[i] + relativedelta(months=month)) < end_date:
                    indicies = month_data.index.intersection(
                        pd.date_range(
                            dates[i] + relativedelta(months=month), 
                            dates[i] + relativedelta(months=month+1) 
                                        - relativedelta(days=1), 
                            freq='d'))
                    
                    month_data.loc[indicies] = \
                                month_data.loc[indicies] \
                                * (overall_data.iloc[month_num][keyword].mean() \
                                / month_data.loc[indicies].mean())

                month_num += 1

            # add data to trend_dict then delete for mem purposes
            trend_dict[keyword] = pd.concat(
                        [trend_dict[keyword], month_data], 
                        axis=0)
            del month_data
        del overall_data

        # fix up DataFrame
        trend_dict[keyword] = trend_dict[keyword].drop('isPartial', axis=1)
        trend_dict[keyword] = trend_dict[keyword].reset_index(level=0)
        trend_dict[keyword] = trend_dict[keyword].rename(
            columns={'date': 'Date', keyword:keywords[word_id]})
        trend_dict[keyword] = trend_dict[keyword].set_index("Date")

        word_id+=1

    df_trends = pd.concat(trend_dict, axis=1)
    df_trends.columns = df_trends.columns.droplevel(0) #drop outside header

    df_trends = df_trends.loc[pd.date_range(start_date, end_date, freq='d')].astype(float32)
    return df_trends