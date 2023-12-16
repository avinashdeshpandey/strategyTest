
# D:\myCompanies_work\55ip\Implementation

# monrets => returns OBTAINED for the month, i.e if you keep your money invested for the ENTIRE THAT MONTH!!!

import os
import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
# Importing packages
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from scipy.optimize import minimize




#data = yf.download('QNIFTY.NS', '2015-01-01')

# data_MF103178 = yf.download('MF103178', '2015-01-01')
# data_MF113047 = yf.download('MF113047', '2015-01-01')

# UDF

def highlight_max(s):
    if s.dtype == np.object:
        is_neg = [False for _ in range(s.shape[0])]
    else:
        is_neg = s < 0
    return ['color: red;' if cell else 'color:black'
            for cell in is_neg]



# this uses nse downloaded data file to calculate monrets
def nsemonthlydaily_pricesNreturns(sec,s_date, e_date, use_eom_datee_to_calculate_monrets=True, price_data_url=''):

    # daily prices
    # lets calcuate monthly returns based on dailyPrices downlaoded
    # read the lastdailyPrice data
    dailyPrices_nse_etf = pd.read_csv(price_data_url)


    # drop unnamed columns
    dailyPrices_nse_etf.drop('Unnamed: 0', axis=1, inplace=True)

    # rename column ClosePrice to Close
    dailyPrices_nse_etf.rename(columns={'ClosePrice': 'Close'}, inplace=True)

    # Filter out only required securities
    dailyPrices_nse_etf = dailyPrices_nse_etf[dailyPrices_nse_etf['Symbol'].isin(sec)]


    # create a copy
    dailyPrices = dailyPrices_nse_etf.copy()


    # format date
    dailyPrices['Date'] = dailyPrices['Date'].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))



    # pull-back sdate by a month
    sdate = datetime.strptime(s_date, "%d-%m-%Y")
    new_s_date = sdate - timedelta(days=30)
    #new_s_date = sdate

    # & extend edates by a month
    edate = datetime.strptime(e_date, "%d-%m-%Y")
    new_e_date = edate + timedelta(days=30)
    new_e_date = edate


    # sdate = datetime.strptime(s_date, "%Y-%m-%d")
    # prev_year = sdate.year - 1
    # lst_s_date = s_date.split(sep="-")
    # lst_s_date[0] = str(prev_year)
    # new_s_date =lst_s_date[0]+"-"+lst_s_date[1]+"-"+lst_s_date[2]
    #
    # edate = datetime.strptime(e_date, "%Y-%m-%d")
    # next_year = edate.year + 1
    # lst_e_date = e_date.split(sep="-")
    # lst_e_date[0] = str(next_year)
    # new_e_date =lst_e_date[0]+"-"+lst_e_date[1]+"-"+lst_e_date[2]


    s_date = new_s_date.date()
    e_date = new_e_date.date()


    if use_eom_datee_to_calculate_monrets==True:
        # DOWNLOAD DAILY PRICES >>> reset-index >>> add 'year_mon
        #prices_d = yf.download(sec, s_date, e_date)
        prices_d = dailyPrices.copy()

        #filter out date range
        prices_d = prices_d[ (prices_d['Date'] >= new_s_date) & (prices_d['Date'] <= new_e_date) ]
        #prices_d = prices_d[(prices_d['Date'] >= sdate.date()) & (prices_d['Date'] <= edate.date())]

        if len(prices_d['Symbol'].unique())!=len(sec):

            monrets = pd.DataFrame()
            only_prices=pd.DataFrame()


        else:

            only_prices = prices_d.copy()

            # RESET INDEX
            #prices_d = prices_d.reset_index()
            # ADD YEAR_MON varibale to locate month-year
            prices_d['year_mon'] = prices_d['Date'].apply(lambda x_date: x_date.strftime("%Y%m"))

            # FILTER required columns only
            #prices_d = prices_d[['Date','Symbol','Close']]

            # FILTER OUT MONTHLY START PRICES
            last_dates_of_the_month_for_trading = prices_d.groupby('year_mon')['Date'].max()

            # FILTER columns & dates only
            prices_d = prices_d[prices_d['Date'].isin(last_dates_of_the_month_for_trading)][['Date','Symbol','Close']]

            # converting data from long-to-wide format using pivot-function
            pricesEOM = prices_d.pivot(index='Date', columns='Symbol', values='Close')
            # pricesEOM = pricesEOM.reset_index()
            # pricesEOM = pricesEOM.set_index('Date')

            # remove commas from the numbers if any
            for x in sec:
                pricesEOM[x] = pricesEOM[x].str.replace(',', '')



            # if pricesEOM.index[0].day < 10:
            #     # This is not a 1st week trading day ..so we will drop this observation
            #     pricesBOM = pricesEOM.iloc[1:]  # ensuring the 1st obse starts on 1st day of the month

            # lets calculate monthly-returns now
            monrets = pricesEOM.astype('float').pct_change()
            monrets = monrets.dropna()
            # monrets => returns OBTAINED for the month, i.e if you keep your money invested for the ENTIRE THAT MONTH!!!

            # change index to YEARMON
            monrets = monrets.reset_index()
            # ADD YEAR_MON varibale to locate month-year
            monrets['Date'] = monrets['Date'].apply(lambda x_date: x_date.strftime("%Y%m"))
            monrets = monrets.set_index('Date')

            #monrets.rename(columns={"Close": sec}, inplace=True)

    return monrets, only_prices






def nsemonthlydaily_pricesNreturns2(sec,s_date, e_date, use_eom_datee_to_calculate_monrets=True, price_data_url=''):
    """
    this an improved version of nsemonthlydaily_pricesNreturns

    we just use masterdailyprices_clean.csv & mastermonrets_clean.csv datasets which have been already created using 2_formreturnsdatabase.py


    :param sec:
    :param s_date:
    :param e_date:
    :param use_eom_datee_to_calculate_monrets:
    :param price_data_url:
    :return:
    """
    # daily prices
    # lets calcuate monthly returns based on dailyPrices downlaoded
    # read the lastdailyPrice data
    if 'gmasterdailyprices' in globals():
        print("Objects loaded")
    else:
        global gmasterdailyprices
        gmasterdailyprices = pd.read_csv(price_data_url + "masterdailyprices_clean.csv")
        gmasterdailyprices.drop('Unnamed: 0', axis=1, inplace=True)


    if 'gmastermonreturns' in globals():
        print("Objects loaded")
    else:
        global gmastermonreturns
        gmastermonreturns = pd.read_csv(price_data_url + "mastermonrets_clean.csv")
        gmastermonreturns.drop('Unnamed: 0', axis=1, inplace=True)






    # filter based on given dates
    # gmasterdailyprices = gmasterdailyprices[(gmasterdailyprices['Date'] >= int(datetime.strftime(datetime.strptime(s_date, "%d-%m-%Y"), "%Y%m"))) &
    #                                     (gmasterdailyprices['Date'] <= int(datetime.strftime(datetime.strptime(e_date, "%d-%m-%Y"), "%Y%m")))]

    subset_masterdailyprices = gmasterdailyprices[ (gmasterdailyprices['Date'] >= datetime.strftime(datetime.strptime(s_date, "%d-%m-%Y"), "%Y-%m-%d")) &
                                             (gmasterdailyprices['Date'] <= datetime.strftime(datetime.strptime(e_date, "%d-%m-%Y"), "%Y-%m-%d"))]




    subset_mastermonreturns = gmastermonreturns[(gmastermonreturns['Date'] >= int(datetime.strftime(datetime.strptime(s_date, "%d-%m-%Y"), "%Y%m"))) &
                                        (gmastermonreturns['Date'] <= int(datetime.strftime(datetime.strptime(e_date, "%d-%m-%Y"), "%Y%m")))]



    # reset index
    subset_mastermonreturns = subset_mastermonreturns.set_index('Date')
    subset_masterdailyprices = subset_masterdailyprices.set_index('Date')

    # filter out columns
    subset_mastermonreturns = subset_mastermonreturns[sec]
    subset_masterdailyprices = subset_masterdailyprices[subset_masterdailyprices['Symbol'].isin(sec)]

    # # remove with 0
    # mastermonreturns = mastermonreturns[~(mastermonreturns == 0).all(axis=1)]
    # mastermonprices = mastermonprices[~(mastermonprices == 0).all(axis=1)]


    return subset_mastermonreturns, subset_masterdailyprices




# this uses yahoo prices directly to calculate monrets
def yfmonthlydaily_pricesNreturns(sec,s_date, e_date, use_eom_datee_to_calculate_monrets=True):

    # pull-back sdate by a month
    sdate = datetime.strptime(s_date, "%Y-%m-%d")
    new_s_date = sdate - timedelta(days=30)

    # & extend edates by a month
    edate = datetime.strptime(e_date, "%Y-%m-%d")
    new_e_date = edate + timedelta(days=30)


    # sdate = datetime.strptime(s_date, "%Y-%m-%d")
    # prev_year = sdate.year - 1
    # lst_s_date = s_date.split(sep="-")
    # lst_s_date[0] = str(prev_year)
    # new_s_date =lst_s_date[0]+"-"+lst_s_date[1]+"-"+lst_s_date[2]
    #
    # edate = datetime.strptime(e_date, "%Y-%m-%d")
    # next_year = edate.year + 1
    # lst_e_date = e_date.split(sep="-")
    # lst_e_date[0] = str(next_year)
    # new_e_date =lst_e_date[0]+"-"+lst_e_date[1]+"-"+lst_e_date[2]


    s_date = new_s_date.date()
    e_date = new_e_date.date()

    # # pull-back sdate & extend edates by a year
    # sdate = datetime.strptime(s_date, "%Y-%m-%d")
    # prev_year = sdate.year - 1
    # lst_s_date = s_date.split(sep="-")
    # lst_s_date[0] = str(prev_year)
    # new_s_date =lst_s_date[0]+"-"+lst_s_date[1]+"-"+lst_s_date[2]
    #
    # edate = datetime.strptime(e_date, "%Y-%m-%d")
    # next_year = edate.year + 1
    # lst_e_date = e_date.split(sep="-")
    # lst_e_date[0] = str(next_year)
    # new_e_date =lst_e_date[0]+"-"+lst_e_date[1]+"-"+lst_e_date[2]
    #
    #
    # s_date = new_s_date
    # e_date = new_e_date

    if use_eom_datee_to_calculate_monrets==True:
        # DOWNLOAD DAILY PRICES >>> reset-index >>> add 'year_mon
        prices_d = yf.download(sec, s_date, e_date)
        # RESET INDEX
        prices_d = prices_d.reset_index()
        # ADD YEAR_MON varibale to locate month-year
        prices_d['year_mon'] = prices_d['Date'].apply(lambda x_date: x_date.strftime("%Y%m"))
        # FILTER OUT MONTHLY START PRICES
        first_date_of_the_month_for_trading = prices_d.groupby('year_mon')['Date'].max()

        # FILTER OUT MONTH PRICES ONLY
        pricesEOM = prices_d[prices_d['Date'].isin(first_date_of_the_month_for_trading)][['Date', 'Adj Close']]
        pricesEOM = pricesEOM.set_index('Date')

        # if pricesEOM.index[0].day < 10:
        #     # This is not a 1st week trading day ..so we will drop this observation
        #     pricesBOM = pricesEOM.iloc[1:]  # ensuring the 1st obse starts on 1st day of the month

        # lets calculate monthly-returns now

        monrets = pricesEOM.pct_change()
        monrets = monrets.dropna()
        # monrets => returns OBTAINED, if you keep your money invested for the ENTIRE THAT MONTH!!!

        # change index to YEARMON
        monrets = monrets.reset_index()
        # ADD YEAR_MON varibale to locate month-year
        monrets['Date'] = monrets['Date'].apply(lambda x_date: x_date.strftime("%Y%m"))
        monrets = monrets.set_index('Date')

        #monrets.rename(columns={"Adj Close": sec}, inplace=True)

    else:
        # DOWNLOAD DAILY PRICES >>> reset-index >>> add 'year_mon
        prices_d = yf.download(sec, s_date, e_date)
        # RESET INDEX
        prices_d = prices_d.reset_index()
        # ADD YEAR_MON varibale to locate month-year
        prices_d['year_mon'] = prices_d['Date'].apply(lambda x_date: x_date.strftime("%Y%m"))
        # FILTER OUT MONTHLY START PRICES
        first_date_of_the_month_for_trading = prices_d.groupby('year_mon')['Date'].min()


        # FILTER OUT MONTH PRICES ONLY
        pricesBOM = prices_d[prices_d['Date'].isin(first_date_of_the_month_for_trading)][['Date','Adj Close']]
        pricesBOM = pricesBOM.set_index('Date')

        # if pricesBOM.index[0].day > 10:
        #     # This is not a 1st week trading day ..so we will drop this observation
        #     pricesBOM = pricesBOM.iloc[1:]             # ensuring the 1st obse starts on 1st day of the month


        # lets calculate monthly-returns now


        monrets = pricesBOM.pct_change()
        monrets = monrets.shift(periods=-1)         # post pct_change , the 1st observation becomes NA, we shift all the observations by -1 coz that reflects returns obtained for the month invested
        monrets = monrets.dropna()                  # Post shift-1 , the last obs becomes NA. we remove that completely
                                                    # monrets => returns OBTAINED, if you keep your money invested for the ENTIRE THAT MONTH!!!

        # change index to YEARMON
        monrets = monrets.reset_index()
        # ADD YEAR_MON varibale to locate month-year
        monrets['Date'] = monrets['Date'].apply(lambda x_date: x_date.strftime("%Y%m"))
        monrets = monrets.set_index('Date')

        monrets.rename(columns={"Adj Close": sec},inplace=True)

    return monrets


def get_rebalance_event_indicator_indices(rebalance='monthly',nobservations=0):
    # time-indices to capture rebalance & cashflow events
    lst_all_indices = [*range(0, nobservations, 1)]

    if rebalance == 'none': # buy & hold
        lst_rebalance_index_binaries = nobservations * [0]

    if rebalance == 'quarterly':
        lst_indices = lst_all_indices[::3]                      # quarter indices
        lst_rebalance_index_binaries = nobservations * [0]

        for indx in lst_indices:
            if indx >= 0:
                # set this indx value to 1 in lst_rebalance_index_binaries
                lst_rebalance_index_binaries[indx] = 1

    if rebalance == 'monthly':
        lst_indices = lst_all_indices[::1]                      # monthly indices
        lst_rebalance_index_binaries = nobservations * [0]

        # if fixed_contribution > 0:
        for indx in lst_indices:
            if indx >= 0:
                lst_rebalance_index_binaries[indx] = 1


    if rebalance == 'annually':
        lst_indices = lst_all_indices[::12]                      # monthly indices
        lst_rebalance_index_binaries = nobservations * [0]

        # if fixed_contribution > 0:
        for indx in lst_indices:
            if indx >= 0:
                lst_rebalance_index_binaries[indx] = 1

    return lst_rebalance_index_binaries

def get_cashflow_contributions(fixed_contribution_frequency='monthly', fixed_contribution=0, nobservations=0):
    # time-indices to capture rebalance & cashflow events
    lst_all_indices = [*range(0, nobservations, 1)]

    if fixed_contribution_frequency == "none":
        lst_contribution_index_binaries = nobservations * [0]

    if fixed_contribution_frequency == "monthly":
        lst_indices = lst_all_indices[::1]

        lst_contribution_index_binaries = nobservations * [0]                    # all 0s

        # if fixed_contribution > 0:
        for indx in lst_indices:                                                    # loop through and wherever indx >=0 fill with fixed_contribution value
            if indx >= 0:
                lst_contribution_index_binaries[indx - 1] = fixed_contribution      # reduced index coz we assume contribution is made end of the previous period and the new amout is then invested BOP

    if fixed_contribution_frequency == 'quarterly':
        lst_indices = lst_all_indices[::3]

        lst_contribution_index_binaries = nobservations * [0]

        for indx in lst_indices:
            if indx >= 0:
                # set this indx value to 1 in lst_rebalance_index_binaries
                lst_contribution_index_binaries[indx - 1] = fixed_contribution

    if fixed_contribution_frequency == 'annually':
        lst_indices = lst_all_indices[::12]

        lst_contribution_index_binaries = nobservations * [0]

        for indx in lst_indices:
            if indx >= 0:
                # set this indx value to 1 in lst_rebalance_index_binaries
                lst_contribution_index_binaries[indx - 1] = fixed_contribution

    return lst_contribution_index_binaries


def backtest(url_path='./', V_0 = 1, rebalance='monthly', fixed_contribution=0 ,monrets="",weights=""):
    # BACK-TEST RESULT OBJECTS: CREATION & INITIALIZATION
    # INITIALIZE : BOP & EOP value arrays/matrix
    bop_value = np.zeros((monrets.shape[0], monrets.shape[1]))
    eop_value = np.zeros((monrets.shape[0], monrets.shape[1]))
    bop_weights = np.zeros((monrets.shape[0], monrets.shape[1]))
    eop_weights = np.zeros((monrets.shape[0], monrets.shape[1]))
    pf_value = np.zeros((monrets.shape[0], 1))

    # INITIALIZE : contribution
    contribution = np.zeros((monrets.shape[0], monrets.shape[1]))

    # SET REBALANCE LIST VALUES
    lst_rebalance_index_binaries = get_rebalance_event_indicator_indices(rebalance, monrets.shape[0])

    # loop through each time-index
    for t in range(len(lst_rebalance_index_binaries)):
        if t==0: # DEFAULT REBALANCE EVENT, ALWAYSSSS!!!
            bop_value[t,:] = weights * V_0     # V_0 can be called as previous EOP value as well. Formula is >>> weights * initial_balance = value invested in corresponding asset at this time!

            # style-1 >>> to emulate https://www.portfoliovisualizer.com/backtest-portfolio#analysisResults
            # eop_value[t, :] = ( (1 + np.array(monrets.iloc[t])) * bop_value[t, :] ) + ( weights * lst_contribution_index_binaries[t] )

            # # style-2 >>> what i would do in practice
            # eop_value[t, :] = ((1 + np.array(monrets.iloc[t])) * bop_value[t, :])

            # style - 3 >>> ultimate final
            eop_value[t, :] = ((1 + np.array(monrets.iloc[t])) * bop_value[t, :])


        else:

            if lst_rebalance_index_binaries[t]==0:  # NO REBALANCE i.e just HOLD & CONTINUE
                #print("NOOOOOOOOOOOOOOOOO")
                bop_value[t, :] = eop_value[t - 1,:]                                          # BOP at t  = EOP at t-1
                eop_value[t, :] = (1 + np.array(monrets.iloc[t])) * bop_value[t, :]           # (1+Rt) * BOP at t


            else: # REBALANCE event!!!
                # STYLE-1 >>> to emulate https://www.portfoliovisualizer.com/backtest-portfolio#analysisResults
                # bop_value[t, :] = (weights * eop_value[t - 1,].sum())
                # eop_value[t, :] = ( (1 + np.array(monrets.iloc[t])) * bop_value[t, :] ) + ( weights * lst_contribution_index_binaries[t] )

                # style-2 >>> what i would do in practice
                # bop_value[t, :] = ( weights * eop_value[t-1,].sum() ) + ( weights * lst_contribution_index_binaries[t] )
                # eop_value[t, :] = ((1 + np.array(monrets.iloc[t])) * bop_value[t, :])

                # style - 3 >>> ultimate final
                bop_value[t, :] = (weights * eop_value[t - 1,].sum())
                eop_value[t, :] = ((1 + np.array(monrets.iloc[t])) * bop_value[t, :])


    # CALCULATE: ASSET WEIGHTS
    for x in range(bop_weights.shape[0]):
        bop_weights[x,:] = bop_value[x,] / bop_value[x].sum()
        eop_weights[x,:] = eop_value[x,] / eop_value[x].sum()

    # FORMATTING: decimal rounding off
    eop_weights = eop_weights.round(7)
    bop_weights = bop_weights.round(7)

    # CALCUATE : ASSET CONTRIBUTIONS
    for t in range(contribution.shape[0]):
        contribution[t, :] = (eop_value[t,] - bop_value[t,]) / sum(bop_value[t,])



    # CALCULATE: PORTFOLIO RETURNS
    R_P = np.sum(eop_value,axis=1).tolist()      # + ( weights * lst_contribution_index_binaries[t] ) for fixed contribution PF value calculations
    R_P.insert(0, V_0)
    R_P = pd.DataFrame(R_P, columns=['PF_RETURNS'])
    R_P = R_P.pct_change()
    R_P.dropna(axis=0,inplace=True)
    R_P = R_P.set_index(monrets.index)

    # CALCULATE: PORTFOLIO BALANCE FOR VARIOUS contribution-frequencies:

    if fixed_contribution==0:
        lst_contri_frequencies = ['none']

    else:
        lst_contri_frequencies = ['none','monthly','quarterly','annually']

    # PF-BALANCE
    PF_BALANCE = pd.DataFrame()
    for contribution_freq in lst_contri_frequencies:

        lst_contribution_index_binaries = get_cashflow_contributions(contribution_freq, fixed_contribution, monrets.shape[0])

        pf_balance=[]
        for i in range(len(lst_contribution_index_binaries)):
            if i==0:
                pf_returns_for_ith_period = 1 + R_P.iloc[i][0]
                pf_bal_ith_eop = ( V_0 * pf_returns_for_ith_period ) + lst_contribution_index_binaries[i]
            else:
                pf_returns_for_ith_period = 1 + R_P.iloc[i][0]
                pf_bal_ith_eop = pf_balance[i-1] * pf_returns_for_ith_period + lst_contribution_index_binaries[i]

            pf_balance.append(pf_bal_ith_eop)

        PF_B = pd.DataFrame(np.array(pf_balance).round(2),columns=['CONTRI_' + contribution_freq])
        PF_BALANCE = pd.concat([PF_BALANCE,PF_B], axis=1)

    ##############################################
    # SAVE ALL RESULTS ...
    ################################################
    # PF-RETURNS
    pdf_pf_returns = pd.DataFrame(np.array(R_P).round(5), columns=['pf_returns_' + rebalance])
    pdf_pf_returns.set_index(R_P.index, inplace=True)
    pdf_pf_returns.to_csv(url_path + "/pdf_pf_returns.csv")

    # PF-BALANCES FOR diff contributions
    PF_BALANCE.set_index(R_P.index, inplace=True)
    PF_BALANCE.to_csv(url_path + "/pdf_pf_balance.csv")

    # WEIGHTS, BALANCES & CONTRIBUTIONS accross assets
    pd.DataFrame(bop_weights).to_csv(url_path + "/bop_weights.csv")
    pd.DataFrame(eop_weights).to_csv(url_path + "/eop_weights.csv")
    pd.DataFrame(bop_value).to_csv(url_path + "/bop_value.csv")
    pd.DataFrame(eop_value).to_csv(url_path + "/eop_value.csv")
    pd.DataFrame(contribution).to_csv(url_path + "/contribution.csv")

    return pdf_pf_returns, PF_BALANCE





# def Reverse(lst):
#     new_lst = lst[::-1]
#     return new_lst


def fgetAnnualized_portfolioNsecurities_summary_forlistofyears(r1, r2, monrets, lst_years):
    """
    :param r1: pf-returns (backtest output
    :param monrets:  monthly security returns
    :param lst_years: list of years for which summaries are sought
    :return: pdf_summary_yearly
    """

    # to save geometric means
    # save portfolio yearly-returns
    lst_pf_returns_yearwise = []
    lst_pf_balance_yearwise = []
    # save monrets yearly-returns
    pdf_monrets_yearly = pd.DataFrame()

    # loop through each year , filter out records and calculate yearly-stats for PF-retunrs & monrets
    for i in range(0, len(lst_years)):
        #print(i)
        """
        for each year:
            filter out returns from r1, r2 & monrets
            calculate geometric mean

        """
        if i == 0:
            # get pf-return data
            r1_subset = r1[r1.index < str(int(lst_years[i]) + 1)]
            # get pf-balance data
            r2_subset = r2[r2.index < str(int(lst_years[i]) + 1)]
            # get monrets data
            monrets_subset = monrets[monrets.index < str(int(lst_years[i]) + 1)]

            # pf-return
            meanG = (np.prod(r1_subset.to_numpy() + 1)) - 1
            # pf-balance -take max
            lastvalue = r2_subset[-1]
            # monrets
            pdf_monrets_meanG = monrets_subset.apply(lambda x: (np.prod(x.to_numpy() + 1)) - 1)
            pdf_monrets_meanG = pd.DataFrame(pdf_monrets_meanG.reset_index(drop=True))
            pdf_monrets_meanG.columns = [lst_years[i]]



        else:
            # get pf data
            r1_subset = r1[(r1.index > str(int(lst_years[i - 1]) + 1)) & (r1.index < str(int(lst_years[i]) + 1))]

            # get pf - balance
            r2_subset = r2[(r2.index > str(int(lst_years[i - 1]) + 1)) & (r2.index < str(int(lst_years[i]) + 1))]
            lastvalue = r2_subset[-1]
            # get monrets
            monrets_subset = monrets[
                (monrets.index > str(int(lst_years[i - 1]) + 1)) & (monrets.index < str(int(lst_years[i]) + 1))]

            # pf
            meanG = (np.prod(r1_subset.to_numpy() + 1)) - 1
            # monrets
            pdf_monrets_meanG = monrets_subset.apply(lambda x: (np.prod(x.to_numpy() + 1)) - 1)
            pdf_monrets_meanG = pd.DataFrame(pdf_monrets_meanG.reset_index(drop=True))
            pdf_monrets_meanG.columns = [lst_years[i]]

        # save all the data

        # save pf returns yearly
        lst_pf_returns_yearwise.append(meanG)

        # save pf balance yearly
        lst_pf_balance_yearwise.append(lastvalue)

        # save monrets yearly-returns
        pdf_monrets_yearly = pd.concat([pdf_monrets_yearly, pdf_monrets_meanG], axis=1)


    # Finally conbine all the data in 1-dataframe
    # get the years-list in a dataframe
    pdf_years = pd.DataFrame({"Year": lst_years})

    # get the pf-yearly returns in a dataframe
    pdf_pf_return_yearly = pd.DataFrame({"PF_return": lst_pf_returns_yearwise})

    # get the pf-yearly returns in a dataframe
    pdf_pf_balance_yearly = pd.DataFrame({"PF_balance": lst_pf_balance_yearwise})


    # lets calculate the cumulative PF_balance
    #pdf_pf_balance_yearly = pd.DataFrame({"PF_balance": V_0 * (pdf_pf_return_yearly['PF_return'] + 1).cumprod()})

    # pdf_monrets_yearly = pdf_monrets_yearly.reset_index()
    pdf_monrets_yearly['Symbol'] = monrets.columns.to_list()
    pdf_monrets_yearly = pd.DataFrame(pdf_monrets_yearly.transpose())

    pdf_monrets_yearly.columns = pdf_monrets_yearly.iloc[pdf_monrets_yearly.shape[0] - 1]

    pdf_monrets_yearly = pdf_monrets_yearly.drop('Symbol')

    pdf_monrets_yearly = pdf_monrets_yearly.reset_index()

    pdf_monrets_yearly.rename(columns={"index": "Year"}, inplace=True)

    pdf_monrets_yearly.reset_index(drop=True, inplace=True)

    # remove column numbers
    # pdf_monrets_yearly.columns = [''] * len(pdf_monrets_yearly.columns)
    # # rename columns
    # pdf_monrets_yearly.columns = Reverse(sec)
    # # drop index
    # pdf_monrets_yearly.reset_index(drop=True, inplace=True)

    # combine pdf_years,pdf_pf_return_yearly and pdf_monrets_yearly
    pdf_summary_yearly = pd.concat([pdf_monrets_yearly, pdf_pf_return_yearly, pdf_pf_balance_yearly], axis=1)

    return pdf_summary_yearly


def fcalculate_PF_performance_measures(pdf_summary_yearly, rf, V_0, fixed_contribution):
    """

    :param pdf_summary_yearly:
    :return:
    """

    # Max drawdown
    wealth_index = V_0 * (pdf_summary_yearly['PF_return'] + 1).cumprod()
    # wealth_index = V_0 * (r1['pf_returns_monthly'] + 1).cumprod()

    previous_peaks = wealth_index.cummax()
    # Plot the previous peaks
    # previous_peaks.plot.line()

    # Calculate the drawdown in percentage
    drawdown = (wealth_index - previous_peaks) / previous_peaks
    # Plot the drawdown
    # drawdown.plot.line()

    # drawdown.min()

    # running_max = np.maximum.accumulate(cumulative)
    # dd_series = (cumulative - running_max) / running_max
    # np.min(dd_series)

    Initial_Balance = V_0
    Final_Balance = pdf_summary_yearly['PF_balance'].to_list()[-1]
    SIP = fixed_contribution
    CAGR = (Final_Balance / Initial_Balance) ** (1 / pdf_summary_yearly['PF_balance'].shape[0]) - 1
    TWRR = (pdf_summary_yearly['PF_return'] + 1).prod().sum() ** (1 / pdf_summary_yearly['PF_balance'].shape[0]) - 1

    mean_yearly = pdf_summary_yearly['PF_return'].mean()
    meanG = (np.prod(pdf_summary_yearly['PF_return'].to_numpy() + 1)) ** (
                1 / (pdf_summary_yearly['PF_return'].shape[0])) - 1
    Stdev = pdf_summary_yearly['PF_return'].std()
    Best_Year = pdf_summary_yearly['PF_return'].max()
    Worst_Year = pdf_summary_yearly['PF_return'].min()


    # drawdown
    semi_deviations = pdf_summary_yearly['PF_return'][pdf_summary_yearly['PF_return'] - mean_yearly < 0]
    semi_dev = np.sqrt(np.square(semi_deviations).sum() / pdf_summary_yearly['PF_return'].shape[0])

    max_drawdown = drawdown.min()

    Sharpe_Ratio = (mean_yearly - rf) / Stdev
    Sortino_Ratio = (mean_yearly - rf) / semi_dev

    # final dataframe
    measure = ["Initial_Balance", "Final_Balance", "SIP", "CAGR", "TWRR", "Stdev", "Best_Year", "Worst_Year", "max_drawdown",
               "Sharpe_Ratio", "Sortino_Ratio"]
    value = [Initial_Balance, Final_Balance, SIP, CAGR, TWRR, Stdev, Best_Year, Worst_Year, max_drawdown, Sharpe_Ratio,
             Sortino_Ratio]

    pdf_performance_measures_based_on_yearwise_summary = pd.DataFrame({"measure": measure,
                                                                       "value": value})

    return pdf_performance_measures_based_on_yearwise_summary


def pltwealthgrowth(ts_port_ret, ts_bmk_ret , ts_port_val, ts_bmk_value,unit="monthly"):
    plt.style.use('ggplot')

    ## re-assign to variables
    r1_pf = ts_port_ret
    r1_bmk = ts_bmk_ret
    r2_pf = ts_port_val
    r2_bmk = ts_bmk_value


    ### PLOTTINGS
    r2_pf = r2_pf.reset_index()
    r2_bmk = r2_bmk.reset_index()
    pdf_portfolio_bmk_values = r2_pf.merge(r2_bmk, on="Date")
    r1_pf = r1_pf.reset_index()
    r1_bmk = r1_bmk.reset_index()
    pdf_portfolio_bmk_returns = r1_pf.merge(r1_bmk, on="Date")

    all_combined_dfs = pdf_portfolio_bmk_returns.merge(pdf_portfolio_bmk_values,on="Date")

    ## LETS PLOT RETURNS AND PORTFOLIO VALUES
    # Lets plot the histogram of the returns.

    # fig = plt.figure()
    # ax1 = fig.add_axes([0.1,0.1,0.8,0.8])

    # Define data values
    x = pdf_portfolio_bmk_values['Date']
    y = pdf_portfolio_bmk_values['Portfolio_value']
    z = pdf_portfolio_bmk_values['benchmark_value']

    # Plot a simple line chart
    plt.plot(x, y, 'c', label='Portfolio')

    # Plot another line on the same chart/graph
    plt.plot(x, z, 'y', label='Benchmark')

    plt.xticks(rotation=45)

    # ax1.hist(pdf_portfolio_bmk_returns[["Date", "Portfolio_returns"]])
    # ax1.set_xlabel('')
    # ax1.set_ylabel("Value (Rupees)")
    # ax1.set_title("")
    plt.xlabel("Year-Months")
    plt.ylabel("Amount(Rs)")
    plt.title("Wealth Growth: Portfolio Value vs Benchmark Value")
    plt.legend()

    return plt,all_combined_dfs


def pltprices(all_prices):

    copy_all_prices = all_prices.drop_duplicates()
    all_prices = copy_all_prices.copy()

    myclose1 = all_prices["Close"].apply(lambda x: float(x))
    all_prices["Close1"] = myclose1
    all_prices.drop(columns=["Close"],inplace=True)
    all_prices.rename(columns={"Close1": "Close"},inplace=True)

    pricesALL = all_prices.pivot_table(index='Date', columns='Symbol', values='Close')

    pricesALL = pricesALL.reset_index()

    ## Save these prcies for plotting
    # plt.figure(figsize=(14, 7))
    # Plot the stock prices of Apple, Facebook, and Amazon
    plt.style.use('ggplot')

    y_variables = pricesALL.columns.to_list()
    y_variables = y_variables.remove('Date')

    pricesALL.plot(x='Date', y=y_variables, figsize=(10, 5))

    # Set the x-axis label
    plt.xlabel('Date')

    # Set the y-axis label
    plt.ylabel('Asset Price')

    # Set the title of the plot
    plt.title('Daily Prices: Assets & Benchmark')

    return plt


def pltprices2(all_prices):

    # copy_all_prices = all_prices.drop_duplicates()
    # all_prices = copy_all_prices.copy()
    #
    # myclose1 = all_prices["Close"].apply(lambda x: float(x))
    # all_prices["Close1"] = myclose1
    # all_prices.drop(columns=["Close"],inplace=True)
    # all_prices.rename(columns={"Close1": "Close"},inplace=True)
    #
    # pricesALL = all_prices.pivot_table(index='Date', columns='Symbol', values='Close')
    pricesALL = all_prices.pivot_table(index='Date', columns='Symbol', values='Close')

    pricesALL = pricesALL.reset_index()


    ## Save these prcies for plotting
    # plt.figure(figsize=(14, 7))
    # Plot the stock prices of Apple, Facebook, and Amazon
    plt.style.use('ggplot')

    y_variables = pricesALL.columns.to_list()
    y_variables = y_variables.remove('Date')

    pricesALL.plot(x='Date', y=y_variables, figsize=(10, 5))

    # Set the x-axis label
    plt.xlabel('Date')

    # Set the y-axis label
    plt.ylabel('Asset Price')

    # Set the title of the plot
    plt.title('Daily Prices: Assets & Benchmark')

    return plt



def dailymonthlypricereturnwideformat(all_prices):
    all_prices_new = all_prices.drop_duplicates()
    all_prices = all_prices_new.copy()
    all_prices['Close'] = list(all_prices['Close'].values.astype(float))

    pricesALL_dailywide = all_prices.pivot(index='Date', columns='Symbol', values='Close')  ## daily prices
    pricesALL_dailywide = pricesALL_dailywide.reset_index()

    ## Monthly prices
    pricesALL_monthwide = pricesALL_dailywide.copy()
    pricesALL_monthwide['year_mon'] = pricesALL_dailywide['Date'].apply(lambda x_date: x_date.strftime("%Y%m"))
    last_dates_of_the_month_for_trading = pricesALL_monthwide.groupby('year_mon')['Date'].max()
    pricesALL_monthwide = pricesALL_monthwide[pricesALL_monthwide['Date'].isin(last_dates_of_the_month_for_trading)]
    pricesALL_monthwide.drop(columns=['year_mon'], inplace=True)

    # for data_freq in ['daily','monthly']:

    #### PORTOLFIO OPTIMIZATION
    # daily & monthly price objects
    # print(pricesALL_dailywide.head())
    # print(pricesALL_monthwide.head())

    # USING PRICES: form daily & monthly returns objects

    # daily returns - arithmetic %change method
    dailyrets_wide = pricesALL_dailywide.copy()
    dailyrets_wide = dailyrets_wide.set_index('Date')
    dailyrets_wide = dailyrets_wide.astype('float').pct_change(1)
    dailyrets_wide = dailyrets_wide.dropna()

    # monthly returns
    monrets_wide = pricesALL_monthwide.copy()
    monrets_wide = monrets_wide.set_index('Date')
    monrets_wide = monrets_wide.astype('float').pct_change(1)
    monrets_wide = monrets_wide.dropna()

    ### LOG RETURNS
    # DAILY
    log_dailyrets_wide = pricesALL_dailywide.copy()
    log_dailyrets_wide = log_dailyrets_wide.set_index('Date')
    log_dailyrets_wide = np.log(log_dailyrets_wide / log_dailyrets_wide.shift(1))
    log_dailyrets_wide = log_dailyrets_wide.dropna()

    # MONTHLY
    log_monrets_wide = pricesALL_monthwide.copy()
    log_monrets_wide = log_monrets_wide.set_index('Date')
    log_monrets_wide = np.log(log_monrets_wide / log_monrets_wide.shift(1))
    log_monrets_wide = log_monrets_wide.dropna()

    return pricesALL_dailywide, pricesALL_monthwide, dailyrets_wide, monrets_wide, log_dailyrets_wide, log_monrets_wide


def foptimalportfoliosNef(df, rf, pdf_security, data_unit):
    # optimization functions
    def portret_mean(weights):
        if data_unit == 'daily':
            weights = np.array(weights)
            ret = np.sum(df.mean() * weights) * 252

        return ret

    def portret_sigma(weights):
        if data_unit == 'daily':
            weights = np.array(weights)
            vol = np.sqrt(np.dot(weights.T, np.dot(df.cov() * 252, weights)))

        return vol

    def port_sharpeRatio(weights):
        if data_unit == 'daily':
            sr = (portret_mean(weights) - rf) / portret_sigma(weights)

        return sr

    # OBJECTIVE FUNCTIONS
    # minimize volatility
    def min_vol(weights):
        return portret_sigma(weights)

    # Maximize sharpe ration  >>> minimize negative sharpe ration -SharpeRatio
    def neg_sharpe(weights):
        return port_sharpeRatio(weights) * -1

    # check allocation sums to 1 ----- sum(weights) - 1 = 0
    def check_sum(weights):
        return np.sum(weights) - 1

    # weight bounds: ALL WEIGHTS POSITIVE & BETWEEN 0 to 1
    bounds = tuple((0, 1) for w in df.columns.tolist())
    # constraints
    constraints = ({'type': 'eq', 'fun': check_sum})
    # initial random weights-guess
    init_guess = np.repeat(1 / pdf_security.shape[0], pdf_security.shape[0]).tolist()

    # OPTIMIZATION BEGINS

    # finding PF with maximum sharpe-ratio
    maximize_sharpe = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    # SAVE THE OPTIMUM RESULTS
    if maximize_sharpe.success == True:

        # print(maximize_sharpe.x)
        # optimal portoflio
        opt_sharpe_wts = maximize_sharpe.x
        opt_sharpe_mean = str(portret_mean(opt_sharpe_wts))
        opt_sharpe_vol = str(portret_sigma(opt_sharpe_wts))

        # print("mean =" + str(portret_mean(opt_sharpe_wts)) + ", " +
        #       "vol =" + str(portret_sigma(opt_sharpe_wts)) + ", " +
        #       "Sharpe =" + str(-1 * neg_sharpe(opt_sharpe_wts)))

    else:
        print(" Optimzation-1: Finding max-sharpe portfolio, Failed !")
        quit()

    # optimization - 2 : finding min-vol-portfolio
    minimum_var_portfolio = minimize(min_vol, init_guess, method='SLSQP', bounds=bounds,
                                     constraints=constraints)

    # optimal portoflio
    if minimum_var_portfolio.success == True:

        # print(minimum_var_portfolio.x)

        opt_MVP_wts = minimum_var_portfolio.x
        opt_MVP_mean = str(portret_mean(opt_MVP_wts))
        opt_MVP_vol = str(portret_sigma(opt_MVP_wts))

        # print("mean =" + str(opt_MVP_mean) + ", " +
        #       "vol =" + str(opt_MVP_vol) + ", " +
        #       "Min Var PF Sharpe =" + str((float(opt_MVP_mean) - rf) / float(opt_MVP_vol)))

    else:
        print(" Optimzation-2: Finding min-variance portfolio, Failed !")
        quit()

    # Putting all results in a dataframe and plotting R-R for best portfiolios, user-pf & individual assets
    if maximize_sharpe.success == True and minimum_var_portfolio.success == True:

        # Lets save means, vol for assets & each of the best portfolios calculated to begin with
        lst_assets = []
        lst_means = []
        lst_vols = []

        # individual assets
        lst_assets.extend(df.columns.tolist())
        lst_means.extend(list(df.mean() * 252))
        lst_vols.extend(list(np.sqrt(df.var() * 252)))

        # max_sharpe ratio
        lst_assets.append('max_sharpe')
        lst_means.append(opt_sharpe_mean)
        lst_vols.append(opt_sharpe_vol)

        # MVP
        lst_assets.append('min_variance')
        lst_means.append(opt_MVP_mean)
        lst_vols.append(opt_MVP_vol)

        # user-selected
        lst_assets.append('YOUR_PF')

        # re-order the columns based on given list
        reorder_df = df[pdf_security['sec'].to_list()]
        reorder_weights = np.array(pdf_security['wts'].to_list()) * 0.01

        if data_unit == 'daily':
            # Expected return
            YOUR_PF_mean = np.sum((reorder_df.mean() * np.array(reorder_weights)) * 252)
            lst_means.append(YOUR_PF_mean)
            # Expected volume
            YOUR_PF_vol = np.sqrt(np.dot(reorder_weights.T, np.dot(reorder_df.cov() * 252, reorder_weights)))
            lst_vols.append(YOUR_PF_vol)

        # rounding to 5-decimals only
        f_lst_means = []
        f_lst_vols = []
        f_lst_numbers = []

        for x in lst_means:
            f_lst_means.append(round(float(x), 5))

        for x in lst_vols:
            f_lst_vols.append(round(float(x), 5))

        for x in range(len(lst_assets)):
            f_lst_numbers.append(x + 1)

        df_optimal_results = pd.DataFrame({
            "Asset": lst_assets,
            "Return": f_lst_means,
            "Risk": f_lst_vols,
            "Assetcolor": f_lst_numbers

        })

        # 1) USER PORTFOLIO
        lst_summary_portfolio_name = []
        lst_summary_portfolio_values = []

        lst_summary_portfolio_name.extend(pdf_security['sec'].to_list())
        lst_summary_portfolio_name.append("ret_mean")
        lst_summary_portfolio_name.append("ret_vol")

        lst_summary_portfolio_values.extend(reorder_weights.round(5))
        lst_summary_portfolio_values.append(YOUR_PF_mean.round(5))
        lst_summary_portfolio_values.append(YOUR_PF_vol.round(5))

        pdf_your_pf = pd.DataFrame({
            "KeyIndex": lst_summary_portfolio_name,
            "YOUR_PF": lst_summary_portfolio_values
        })

        # 2) Max sharpe PF
        lst_summary_portfolio_name = []
        lst_summary_portfolio_values = []

        lst_summary_portfolio_name.extend(df.columns.tolist())
        lst_summary_portfolio_name.append("ret_mean")
        lst_summary_portfolio_name.append("ret_vol")

        lst_summary_portfolio_values.extend(opt_sharpe_wts.round(5))
        lst_summary_portfolio_values.append(round(float(opt_sharpe_mean), 5))
        lst_summary_portfolio_values.append(round(float(opt_sharpe_vol), 5))

        pdf_maxsharpe_pf = pd.DataFrame({
            "KeyIndex": lst_summary_portfolio_name,
            "MAX_SHARPE_PF": lst_summary_portfolio_values
        })

        # 2) Min variance PF
        lst_summary_portfolio_name = []
        lst_summary_portfolio_values = []

        lst_summary_portfolio_name.extend(df.columns.tolist())
        lst_summary_portfolio_name.append("ret_mean")
        lst_summary_portfolio_name.append("ret_vol")

        lst_summary_portfolio_values.extend(opt_MVP_wts.round(5))
        lst_summary_portfolio_values.append(round(float(opt_MVP_mean), 5))
        lst_summary_portfolio_values.append(round(float(opt_MVP_vol), ))

        pdf_minvar_pf = pd.DataFrame({

            "KeyIndex": lst_summary_portfolio_name,
            "MIN_VAR_PF": lst_summary_portfolio_values

        })

        pdf_BEST_PFs = pdf_your_pf.merge(pdf_maxsharpe_pf, on="KeyIndex")
        pdf_BEST_PFs = pdf_BEST_PFs.merge(pdf_minvar_pf, on="KeyIndex")

        # PLOT EFFICIENT FRONTIER : we need to find out the various optimum portfolio (MVPs) for given return...we will use
        ## plotting the effiecient portfolios

        lst_ef_r = []
        lst_ef_vols = []
        #  , df_optimal_results['Return'].max()
        for r in np.linspace(float(opt_MVP_mean), float(opt_sharpe_mean), 11):
            # constraints
            constraints_ef = ({'type': 'eq', 'fun': check_sum},
                              {'type': 'eq', 'fun': lambda x: portret_mean(x) - r}

                              )

            ef_vol_for_given_r = minimize(min_vol, init_guess, method='SLSQP', bounds=bounds,
                                          constraints=constraints_ef)

            lst_ef_r.append(r)

            opt_wts = ef_vol_for_given_r.x
            opt_ef_vol = str(portret_sigma(opt_wts))
            lst_ef_vols.append(round(float(opt_ef_vol), 5))

        pdf_ef_rr_values_to_plot = pd.DataFrame({
            "Asset": "",
            "Return": lst_ef_r,
            "Risk": lst_ef_vols,
            "Assetcolor": df_optimal_results.shape[0] + 11

        })

        # combine
        df_optimal_results_plot = pd.concat([df_optimal_results, pdf_ef_rr_values_to_plot], axis=0)

        pdf_sorted_forplot = df_optimal_results_plot.sort_values(by=['Return'])
        lst_va_position = ['top','bottom'] * int(int(pdf_sorted_forplot.shape[0]/2) + 1)
        removeK_items = len(lst_va_position) - pdf_sorted_forplot.shape[0]
        lst_va_position = lst_va_position[: len(lst_va_position) - removeK_items]

        pdf_sorted_forplot['text_va'] = lst_va_position

        plt.style.use('ggplot')
        plt.figure(figsize=(12, 8))
        plt.scatter(pdf_sorted_forplot['Risk'], pdf_sorted_forplot['Return'],
                    c=pdf_sorted_forplot['Assetcolor'])

        # plt.scatter(volatility,returns,c=sharpe)
        # add labels to all points
        for i in range(pdf_sorted_forplot.shape[0]):
            plt.text(pdf_sorted_forplot.iloc[i,]['Risk'], pdf_sorted_forplot.iloc[i,]['Return'],
                     pdf_sorted_forplot.iloc[i,]['Asset'], va=pdf_sorted_forplot['text_va'].tolist()[i], ha='center')

        # plt.scatter(pdf_ef_rr_values_to_plot['Risk'],pdf_ef_rr_values_to_plot['Return'])
        plt.title("PORTFOLIO EFFICIENT FRONTIER")
        plt.xlabel('Volatility')
        plt.ylabel('Return')

        return plt, pdf_BEST_PFs, df_optimal_results

# DATA IMPORT/ DOWNLOAD