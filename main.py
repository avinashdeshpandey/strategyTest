# env mmpy37
# dynamic assest inclsiion code: https://stackoverflow.com/questions/76692907/streamlit-dynamic-form-add-and-remove-fields-dynamically

# with st.form(key='columns_in_form'):
#     c1, c2, c3, c4 = st.columns(4)
#     with c1:
#         initialInvestment = st.text_input("Starting capital",value=500)
#     with c2:
#         monthlyContribution = st.text_input("Monthly contribution (Optional)",value=100)
#     with c3:
#         annualRate = st.text_input("Annual increase rate in percentage",value="15")
#     with c4:
#         investingTimeYears = st.text_input("Duration in years:",value=10)
#
#     submitButton = st.form_submit_button(label = 'Calculate')

# This is the HOME PAGE START FROM THIS PAGE
# For now this is in TESTING MODE:
# 1) line 220 asn indTest=TRUE ..this uses USA tickers and yahoo financials to downlaod prices
# 2) Continue testing for a few other portfolios and
# 3) NEED TO BUILD: Dynamic inclusion of assest rows feature..once thats done the version1 of the app can be deployed!!! :)


# use terminal and this command to use web version: # python -m streamlit run  D:\Projects_inDev\INVESTMENT_DSS\bt_home3_test.py


# security list and db path
URL_UPDATED_DATA_LOC = "./data/downloads/prices/LATEST_UPDATED_DB/"
ETF_LIST_FILE_NAME = "tickunivers.csv"

# #MASTER_ETF_DAILY_PRICE_DB_NAME = "masterALLETF_nsedailyprices_updated_2023_08_31.csv"
# MASTER_ETF_DAILY_PRICE_DB_NAME2 = "mastermonrets.csv"

investor_name = "TEST"

#### MAIN CODE STARTS ----

#try:

import os
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import random
import string
import bt_udf


st.set_page_config(layout="wide")  # this uses all screen space

# """
# we want to give user ability to include as many as asset they want dynamically.
# In order to do that , we use session-state feature as follows;
#
# For each UI element to de added;
# create a session state variable & define all variables and functions to be used for the variable
# e.g; lets say we want to offer ability to dynamically add & delete select-box.
#     1) Declare "field_selectbox" as session_state variable
#     2) For this session state variable add;
#         --> st.session_state.field_selectbox = 0
#         --> st.session_state.addfield_selectbox = []
#         --> st.session_state.deletefields = []
#
#
# """

#########################################################
# SESSION FUNCTIONS / DYNAMIC UI ELEMENTS /  VARIABLES - BEGINS
#######################################################

## session state variables for field_selectbox

## select box -counter
def add_field_selectbox():
    st.session_state.field_selectbox += 1

## seelct box session state declarartion
if "field_selectbox" not in st.session_state:
    st.session_state.field_selectbox = 0
    st.session_state.addfield_selectbox = []
    st.session_state.deletefields = []


## weight box counter
def add_field_weightbox():
    st.session_state.field_weightbox += 1


## field_weightbox session state declarartion
if "field_weightbox" not in st.session_state:
    st.session_state.field_weightbox = 0
    st.session_state.addfield_weightbox = []
    #st.session_state.deletefield_weightbox = []


## a delete function to delete session-state UI elements
def delete_fields(index):
    # Assets
    st.session_state.field_selectbox -= 1

    # remove select-boxes
    del st.session_state.addfield_selectbox[index]
    #del st.session_state.deletefield_selectbox[index]
    del st.session_state.deletefields[index]

    # weights
    st.session_state.field_weightbox -= 1

    # remove select-boxes
    del st.session_state.addfield_weightbox[index]
    #del st.session_state.deletefield_weightbox[index]



with st.sidebar:
    st.write("PORTFOLIO BACKTESTING")
    st.write("1. Select Parameters: Date-range, amounts, frequency")
    st.write("2. Select Securities: Securities, weights")
    st.write("3. Perform Backtest: Assess Returns, final portfolio-value")

    st.write("----------------------------")


#########################################################
# SESSION FUNCTIONS / DYNAMIC UI ELEMENTS /  VARIABLES - BEGINS
#######################################################

####################################################################
# DATA IMPORT
######################################################################
# st.markdown("***")
# st.write("Steps for Building an ETF Portfolio")
# st.write("https://www.investopedia.com/articles/exchangetradedfunds/11/building-an-etf-portfolio.asp")

# st.write("This tool allows you to construct one or more portfolios based on the selected ETFs. "
#          "You can analyze and backtest portfolio returns, risk characteristics, style exposures, and drawdowns."
#          "Backtest a portfolio asset allocation and compare historical and realized returns and risk characteristics against various lazy portfolios")


# SECURITY UNIVERSE -- etf only for now
all_nse_etf_symbols = pd.read_csv(URL_UPDATED_DATA_LOC + ETF_LIST_FILE_NAME,sep=",", encoding='utf-8')
tickers = ['Select Asset']
tickers.extend(all_nse_etf_symbols['Symbol'].unique().tolist())
# tickers.extend(all_nse_stocks_symbols['SYMBOL'].unique().tolist())
# ticks = all_usa_etfs['tick'].apply(lambda x: x.upper())
# ticks = ticks.to_list()
# tickers.extend(ticks)




#st.set_page_config(page_title="Backtest Portfolio Asset Class Allocation", page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)
# start_ymd = st.sidebar.date_input('Start Year', datetime.date(2017,5,1))
# end_ymd = st.sidebar.date_input('End Year', datetime.date(2022,5,1))
st.title("PORTFOLIO ASSESSMENT USING BACKTESTING")
st.write("***ETF/Index-fund based optimal portfolio assessment by way of backtesting method tool***")
st.markdown("***")


with st.container():
    # USER-INPUT SECTION - 2
    col1, colempty2, c2, c3, c4 = st.columns([1,1,1,1,1])

    with col1:
        st.subheader("1. Select Parameters")
        #st.markdown("***")

        Start_yyyy_mm_dd = st.date_input('Start Year (YYYY/MM/DD)', value=datetime.datetime.strptime('2020-04-01',
                                                                                                     '%Y-%m-%d'))  # , min_value = datetime.date(2000,3,1))
        Start_yyyy_mm_dd = Start_yyyy_mm_dd.strftime('%Y-%m-%d')
        end_yyyy_mm_dd = st.date_input('End Year (YYYY/MM/DD)', value=datetime.datetime.strptime('2023-03-31',
                                                                                                 '%Y-%m-%d'))  # , max_value = datetime.date(2030,5,1))
        end_yyyy_mm_dd = end_yyyy_mm_dd.strftime('%Y-%m-%d')
        InitialAmount = st.number_input("Initial Amount", min_value=0.0)
        Rebalancing = st.selectbox('Rebalancing', ('none', 'monthly', 'quarterly', 'annually'))

        Cashflows = st.selectbox('Cashflows', ('none', 'Contribute fixed amount'))

        #print(">>> " + str(Start_yyyy_mm_dd))

        if Cashflows == 'Contribute fixed amount':
            ContributionAmount = st.number_input("Contribution Amount", min_value=0.0)
            Contributionfrequency = st.selectbox('Contribution frequency', ('monthly', 'quarterly', 'annually'))
        else:
            ContributionAmount = 0
            Contributionfrequency = 'none'

        rf = st.number_input("Risk Free Rate(%)", min_value=0.0)   ## riskfree rate
        benchmark_ticker = st.selectbox("Benchmark ", options=tickers)

    with colempty2:
        st.subheader("2. Select Securities")
        st.button("➕ ADD SECURITY", on_click=add_field_selectbox)

    with c2:
        st.markdown("***")
        st.write("***ASSETS***")

    with c3:
        st.markdown("***")
        st.write("***WEIGHTS***")

    with c4:
        st.markdown("***")
        st.write("***DELETE***")




for i in range(st.session_state.field_selectbox):
    with c2:
        #st.session_state.addfield_selectbox.append(st.text_input(f"Field {i}", key=f"text{i}"))
        st.session_state.addfield_selectbox.append(st.selectbox("", options=tickers, key=f"selbox{i}"))


    with c3:
        st.session_state.addfield_weightbox.append(st.number_input("", min_value=0.0, key=f"inputbx{i}"))
        #st.session_state.deletefield_selectbox.append(st.button("❌", key=f"delete{i}", on_click=delete_fields, args=(i,)))


    with c4:
        st.markdown("<p style='padding-top:29px'></p>", unsafe_allow_html=True)
        st.session_state.deletefields.append(
            st.button("DELETE ROW", key=f"delete{i}", on_click=delete_fields, args=(i,)))

st.markdown("***")
st.subheader("3. Perform Backtest")
st.write("***Please make sure that you have selected securities for your 'portfolio' before performing backtesting***")
submitButton = st.button(label='BACKTEST')


if submitButton==True:

    st.subheader("Portfolio Analysis Results (" + str(Start_yyyy_mm_dd) + " " + str(end_yyyy_mm_dd) + ")")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Summary", "Annual Returns", "Monthly Returns", "PF_Growth", "Optimization"])

    ############################################################
    # INPUT CHECK :  ASSET SELECTION
    ############################################################
    #pf1_weights = [col2_1,col2_2]
    sec = []
    wts_1 = []

    ## lets get all the securities and wts entered by user
    for i in range(st.session_state.field_selectbox):
        #st.write(i)
        #st.write(st.session_state["selbox" + str(i)])
        #st.write(st.session_state["inputbx" + str(i)])
        sec.append(st.session_state["selbox" + str(i)])
        wts_1.append(st.session_state["inputbx" + str(i)])


    ### VALUE CHECK

    if len(sec)==0:
        st.write("ASSET SELECTION ERROR: Please select at least 1-ASSET for your portfolio")

    if all(i > 0 for i in wts_1)==False:
        st.write("NEGATIVE / ZERO WEIGHTS ERROR: ALL Assets weights MUST be  > 0 !")

    if sum(wts_1)!=100:
        st.write("WEIGHTS SUM ERROR: Weights must sum to 100")

    # ensure wts sum to 1 & all wts are > 0
    if sum(wts_1)==100 and all(i > 0 for i in wts_1)==True:

        # # # # ## USER INPUT --sample only
        # Start_yyyy_mm_dd = '2014-04-01'
        # end_yyyy_mm_dd = '2023-03-31'
        # InitialAmount = 1
        # Rebalancing = 'monthly'
        # Cashflows = 'Contribute fixed amount'
        # #Cashflows = 'none'
        # ContributionAmount = 1
        # Contributionfrequency = 'monthly' # 'quarterly'
        #
        #
        # rf = 0.0
        # benchmark_ticker = 'NIFTY 50'    #'GOLDBEES'
        #
        # # DEAFULT PORTFOLIO
        # sec = ['TATAMOTORS','L&TFH','MUKANDLTD']   # ['BANKBEES', 'NIFTYBEES', 'JUNIORBEES']
        # wts_1 = [50,25,25]


        ##################################################
        ### RECAST & SET SOME VALUES BASED ON USER INPUT
        ##################################################

        # SETTING SOME VALUES
        bmk = [benchmark_ticker]
        bmk_wts = [100]    # default for now


        if Cashflows == 'Contribute fixed amount':
            ContributionAmount = ContributionAmount
            Contributionfrequency = Contributionfrequency
        else:
            ContributionAmount = ContributionAmount
            Contributionfrequency = Contributionfrequency

        if ContributionAmount == 0:
            Contributionfrequency = "none"
            Contributionfrequency = "none"


        fixed_contribution = ContributionAmount


        # UI-date format recasting
        s_date = datetime.datetime.strftime(datetime.datetime.strptime(Start_yyyy_mm_dd,"%Y-%m-%d"),"%d-%m-%Y")  #Start_yyyy_mm_dd       # Start_yyyy_mm_dd = '2018-04-01'  >>> "%d-%m-%Y")
        e_date = datetime.datetime.strftime(datetime.datetime.strptime(end_yyyy_mm_dd, "%Y-%m-%d"), "%d-%m-%Y")                                   #end_yyyy_mm_dd         # end_yyyy_mm_dd = '2019-04-01'
        #sec = asset_wts['sec'].to_list()

        V_0 = InitialAmount                                     #InitialAmount  # InitialAmount = 1
        rebalance = Rebalancing                                 #'none' #'monthly'   #Rebalancing      #'quarterly'  # monthly, quarterly   Rebalancing = 'monthly'

        #########################################################################################################################
        # PROCESSING BEGINS
        ########################################################################################################################
        # Lets create dataframe for PF - securities & wts & BMK securities & wts separately
        pdf_security = pd.DataFrame({"sec": sec,
                                     "wts": wts_1})

        pdf_bmks = pd.DataFrame({"sec": bmk,
                                 "wts": bmk_wts})



        # asset_wts = pd.DataFrame()
        # asset_wts['sec'] = sec                          #= ['SPY','AGG']      # sec = ['NIFTYBEES','GOLDBEES'] ['HDIL','TCS']
        # asset_wts['pf1_wts'] = wts_1                    #  [50,50]              # wts_1 = [20, 80]
        #
        # asset_wts = asset_wts.groupby('sec').sum()
        # asset_wts = asset_wts.reset_index()


        # lets loop through each pdf at a time and calculate monthly & annual & final summaries
        loop_names = ['PF', 'BMK']

        all_summary1 = pd.DataFrame()
        all_summary2 = pd.DataFrame()

        all_r1s = pd.DataFrame()
        all_r2s = pd.DataFrame()

        all_prices = pd.DataFrame()

        all_securities = []

        all_monrets = pd.DataFrame()

        for lpname in loop_names:

            if lpname == 'PF':
                pf = pdf_security

            if lpname == 'BMK':
                pf = pdf_bmks

            sec = pf['sec'].to_list()
            weights = np.array(pf['wts'].to_list()) * 0.01

            all_securities.extend(sec)

            # CALCULATE RETURNS:
            # if indtest == False:
            #     monrets = bt_udf.yfmonthlydaily_pricesNreturns(sec, s_date, e_date, use_eom_datee_to_calculate_monrets=True)
            # else:

            monrets, only_prices = bt_udf.nsemonthlydaily_pricesNreturns2(sec, s_date, e_date, use_eom_datee_to_calculate_monrets=True,
                                                                price_data_url=URL_UPDATED_DATA_LOC)

            # convert index to str
            monrets = monrets.reset_index()
            monrets['Date'] = monrets['Date'].apply(lambda x: str(x))
            monrets = monrets.set_index('Date')

            # """
            # only_prices >>> 3-columns (Date, Symbol, Close), must be long format, has daily price data and date-format as YYYY-MM-dd
            # """
            # #only_prices = only_prices.reset_index()
            # #only_prices['Date'] = only_prices['Date'].apply(lambda x: str(x))
            # #only_prices = only_prices.set_index('Date')



            if monrets.shape[0]==0 or only_prices.shape[0]==0:

                st.subheader("PRICES FOR SOME OF THE SECURITIES NOT FOUND ..PLEASE CHANGE THE ASSET!!!")
                st.stop()

            else:

                monrets = round(monrets, 6)


                # only_prices = only_prices[['Date','Symbol','Close']]
                #
                # # replace ","
                # only_prices['Close'] = only_prices['Close'].apply(lambda x: str(x).replace(',', ''))
                #
                if all_prices.shape[0] == 0:
                    # save monthly prices
                    #all_prices = only_prices.reset_index()

                    # save all price --no index
                    all_prices = only_prices.reset_index()

                    # save all-returns--no index
                    all_monrets = monrets.reset_index()


                else:

                    pdf_temp = only_prices.reset_index()
                    all_prices = pd.concat([all_prices, pdf_temp], axis=0)

                    # save returns
                    pdf_temp2 = monrets.reset_index()
                    all_monrets = all_monrets.merge(pdf_temp2,on="Date")


                #dailyPrices_nse_etf.rename(columns={'Adj Close': 'Close'}, inplace=True)

                #monrets = bt_udf.monthlydaily_pricesNreturns(sec, s_date, e_date, use_eom_datee_to_calculate_monrets=True, dailyPrices_use_nse_data=True)

                url_main_folder_name = investor_name + "_" + str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

                # CREATE FOLDERS
                if os.path.exists('output/tool/' + url_main_folder_name) == False:
                    os.makedirs('output/tool/' + url_main_folder_name)

                url_path = 'output/tool/' + url_main_folder_name


                r1, r2 = bt_udf.backtest(url_path=url_path, V_0=V_0, rebalance=rebalance, fixed_contribution=fixed_contribution, monrets=monrets, weights=weights)

                min_year = datetime.datetime.strptime(str(r1.index.min()), "%Y%m").year
                max_year = datetime.datetime.strptime(str(r1.index.max()), "%Y%m").year

                # lets get the list of years
                lst_years = []
                for yr in range(min_year, max_year + 1, 1):
                    lst_years.append(yr)

                ## lets get the summaries by year
                if lpname == 'PF':
                    pdf_summary_yearly = bt_udf.fgetAnnualized_portfolioNsecurities_summary_forlistofyears(r1, r2['CONTRI_' + Contributionfrequency], monrets, lst_years)
                    pdf_performance_summary = bt_udf.fcalculate_PF_performance_measures(pdf_summary_yearly, rf, V_0,fixed_contribution)

                    pdf_performance_summary.rename(columns={"value": "PF"}, inplace=True)

                    r1_pf = r1.copy()
                    r1_pf.rename(columns={"pf_returns_" + rebalance: "Portfolio_returns"}, inplace=True)
                    r2_pf = r2[['CONTRI_' + Contributionfrequency]].copy()
                    r2_pf.rename(columns={'CONTRI_' + Contributionfrequency: "Portfolio_value"}, inplace=True)

                if lpname == 'BMK':
                    pdf_summary_yearly = bt_udf.fgetAnnualized_portfolioNsecurities_summary_forlistofyears(r1, r2['CONTRI_' + Contributionfrequency], monrets, lst_years)
                    pdf_performance_summary = bt_udf.fcalculate_PF_performance_measures(pdf_summary_yearly, rf, V_0,
                                                                                 fixed_contribution)

                    pdf_summary_yearly.rename(columns={"PF_return": "BMK_return"}, inplace=True)
                    pdf_summary_yearly.rename(columns={"PF_balance": "BMK_balance"}, inplace=True)
                    pdf_performance_summary.rename(columns={"value": "BMK"}, inplace=True)

                    r1_bmk = r1.copy()
                    r1_bmk.rename(columns={"pf_returns_" + rebalance: "benchmark_returns"}, inplace=True)
                    r2_bmk = r2[['CONTRI_' + Contributionfrequency]].copy()
                    r2_bmk.rename(columns={'CONTRI_' + Contributionfrequency: "benchmark_value"}, inplace=True)



                if all_summary1.shape[0] == 0:

                    all_summary1 = pdf_summary_yearly.copy()
                    all_summary2 = pdf_performance_summary.copy()

                else:
                    all_summary1 = all_summary1.merge(pdf_summary_yearly[["Year", "BMK_return", "BMK_balance"]],
                                                      how='inner', on="Year")
                    all_summary2 = all_summary2.merge(pdf_performance_summary, how='inner', on="measure")


        # POST COMPLETION OF FOR LOOP FOR PF & BMK portfolios...we summarize & plot
        ## SAVE THE RESULTS
        name_order1 = ["Year", "PF_return", "PF_balance", "BMK_return", "BMK_balance"] + all_securities
        subset_name_order1 = []
        for x in name_order1:
            if x != bmk[0]:
                subset_name_order1.append(x)

        # ordering columns
        all_summary1 = all_summary1[subset_name_order1]
        all_summary1.to_csv(url_path + "summary1_PF_n_BMK_yearwise.csv")

        all_summary2.to_csv(url_path + "summary2_PF_n_BMK_measures_using_summary1yearwise_results.csv")
        all_prices.to_csv(url_path + "PF_BMK_ALL_daily_prices.csv")

        # pltwealth, pdfs_all_combined_data = bt_udf.pltwealthgrowth(r1_pf, r1_bmk, r2_pf, r2_bmk)
        #
        # print(pdfs_all_combined_data.columns)

        ## DISPLAY SUMMARY
        with tab1:
            st.write("SUMMARY")
            st.dataframe(all_summary2)

        with tab2:
            st.write("Annual Returns")
            st.dataframe(all_summary1)

        #st.write(df.style.format("{:.2}"))
        #st.markdown("***")


        ## LETS DISPLAY PLOTS
        with tab4:
            # USER-INPUT SECTION - 2
            c5, c6 = st.columns([1, 1])

            with c5:

                ## DISPLAY PLOTS
                st.write("Wealth Growth")
                pltwealth, pdfs_all_combined_data = bt_udf.pltwealthgrowth(r1_pf,r1_bmk,r2_pf,r2_bmk)
                st.pyplot(pltwealth)

                #plt.show()
            with c6:
                ## PRICE PLOTS
                st.write("Asset Prices")
                pltprices = bt_udf.pltprices2(all_prices)
                st.pyplot(pltprices)


        with tab3:

            st.write("Monthly Returns")
            all_monrets.drop(columns=[bmk[0]],inplace=True)
            name_order1 = ["Date", "Portfolio_returns", "Portfolio_value", "benchmark_returns", "benchmark_value"]
            pdfs_all_combined_data = pdfs_all_combined_data[name_order1]
            pdfs_all_combined_data = pdfs_all_combined_data.merge(all_monrets,on="Date")
            st.dataframe(pdfs_all_combined_data)



        # """
        # Optimization part begins from here
        #
        # """
        # Please select data-unit & data-scaling factor
        data_unit = 'daily'  # 'monthly'
        data_scale = 'percentage'  # 'log'


        all_prices['Date'] = all_prices['Date'].apply(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d"))

        ## wide format: daily prices, monthly prices, daily returns, monthly returns, log-daily rets, log monthly rets
        pricesALL_dailywide, pricesALL_monthwide, dailyrets_wide, monrets_wide, log_dailyrets_wide, log_monrets_wide = bt_udf.dailymonthlypricereturnwideformat(all_prices)


        # using pct-change objects (daily returns)
        if bmk[0] in dailyrets_wide.columns:
            # daily data for assets (no index in this datasets)
            subset_dailyrets_wide = dailyrets_wide.drop(columns=[bmk[0]], axis=1)
            log_subset_dailyrets_wide = log_dailyrets_wide.drop(columns=[bmk[0]], axis=1)
            # monthly data for assets (no index in this datasets)
            subset_monrets_wide = monrets_wide.drop(columns=[bmk[0]], axis=1)
            log_subset_monrets_wide = log_monrets_wide.drop(columns=[bmk[0]], axis=1)

        else:
            # daily data for assets (no index in this datasets)
            subset_dailyrets_wide = dailyrets_wide.copy()
            log_subset_dailyrets_wide = log_dailyrets_wide.copy()
            # monthly data for assets (no index in this datasets)
            subset_monrets_wide = monrets_wide.copy()
            log_subset_monrets_wide = log_monrets_wide.copy()

        # select the APPROPRIATE dataset for optimization
        # LETS select the appropriate dataframe based on user entry
        if data_unit == 'daily':
            if data_scale == 'percentage':
                df = subset_dailyrets_wide.copy()
            elif data_scale == 'log':
                df = log_subset_dailyrets_wide.copy()
            else:
                print("optimization dataset is not being selected!")
                quit()

        elif data_unit == 'monthly':
            if data_scale == 'percentage':
                df = monrets_wide.copy()
            elif data_scale == 'log':
                df = log_monrets_wide.copy()
            else:
                print("optimization dataset is not being selected!")
                quit()

        else:
            print(" data-unit or data-scale options not found!")
            quit()


        ## optimization

        plt_rr_bestPFsNef, pdf_BEST_PFs, df_optimal_results = bt_udf.foptimalportfoliosNef(df, rf, pdf_security, data_unit)

        with tab5:
            # USER-INPUT SECTION - 2
            st.subheader("Optimal Portfolios")
            st.pyplot(plt_rr_bestPFsNef)
            # pdf_rr = pdf_BEST_PFs[pdf_BEST_PFs['KeyIndex'].isin(['ret_mean', 'ret_vol'])]
            #
            # ## DISPLAY PLOTS
            # st.subheader("Optimal PFs: Risk & Returns")
            # st.dataframe(pdf_rr.transpose())

            tab5c1, tab5c2 = st.columns([1,1])
            with tab5c1:
                pdf_rr = pdf_BEST_PFs[pdf_BEST_PFs['KeyIndex'].isin(['ret_mean', 'ret_vol'])]

                ## DISPLAY PLOTS
                st.subheader("Optimal Portfolios: Means and Variances")
                #st.dataframe(pdf_rr.transpose())
                pdf_rr.set_index('KeyIndex', inplace=True)
                st.dataframe(pdf_rr)

            with tab5c2:
                st.subheader("Optimal Portfolios: Asset Allocations (weights)")
                pdf_assestweights = pdf_BEST_PFs[~pdf_BEST_PFs['KeyIndex'].isin(['ret_mean', 'ret_vol'])]
                #tpdf_assestweights = pdf_assestweights.transpose()
                # st.pyplot(pdf_assestweights.transpose())
                pdf_assestweights.set_index('KeyIndex', inplace=True)
                st.dataframe(pdf_assestweights)


# except:
#
#     print("An exception occurred")


