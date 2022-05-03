__all__ = [
    "read_nymex",
    "read_google_trends",
]

from calendar import month
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
    for keyword, category in zip(keywords, categories):
        # Find breakpoints for dates not in our dataset yet
        date_not_included = pd.date_range(start_date, 
                                          end_date, 
                                          freq='d').difference(
                                google_trends_df.index)
        dates_with_nan = google_trends_df.loc[
                        (google_trends_df.index >= start_date) 
                        & (google_trends_df.index <= end_date)
                    ][pd.isnull(google_trends_df[keyword])].index

        date_bps = find_breakpoints(date_not_included.union(dates_with_nan))

        # Loop over these dates and fill them in with pytrends
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
                                        [category],
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

    MONTH_CHUNK = 6

    overall_results = []
    word_id = 0
    for keyword, category in individual_exact_keyword:
        current = start_date

        # Fetch monthly google trends
        pytrend.build_payload(kw_list=[keyword], 
                                timeframe=overall_timeframe,
                                geo=country, 
                                cat=category,
                                gprop=search_type)
        monthly = pytrend.interest_over_time()
        monthly = monthly.reindex(
            pd.date_range(start_date, end_date, freq='d')) # Expand to days
        monthly = monthly.ffill().bfill() # Fill NaNs
        time.sleep(5)

        keyword_results = []
        while current <= end_date:
            last_date_of_chunk = min(
                end_date,
                current + relativedelta(months=MONTH_CHUNK) - relativedelta(days=1)
            )
            timeframe = current.isoformat() + " " + last_date_of_chunk.isoformat()
            pytrend.build_payload(
                kw_list=[keyword],
                timeframe=timeframe,
                geo=country,
                cat=category,
                gprop=search_type
            )
            keyword_results.append(pytrend.interest_over_time())
            current = current + relativedelta(months=MONTH_CHUNK)
            time.sleep(5)
            
        # fix up DataFrame
        daily = pd.concat(keyword_results,axis=0)
        daily = daily.drop("isPartial", axis=1)
        daily[keyword] = daily[keyword] * monthly[keyword] / 100
        daily = daily.rename(
            columns={keyword:keywords[word_id]})

        keyword_results.append(daily)
        word_id += 1

    df_trends = pd.concat(keyword_results, axis=1)
    df_trends.index.names = ["Date"]

    df_trends = df_trends.astype(float32)
    return df_trends