"""
Python Functions for LMN Dashboarding Project.

In the future this may be splitted up into multiple files if there are too many functions here

BMO
June 2022 | last update: June 2022
"""

# Function to automatically install packages when not available
import pip

def import_or_install(package):
    try:
        __import__(package)
        print("imported")
    except ImportError:
        pip.main(['install', '--user', package])
        print("installed")

# Installing basic packages for the first time
import_or_install('re')
import_or_install('pandas')
import_or_install('pandas-profiling')
import_or_install('numpy')
import_or_install('seaborn')
import_or_install('plotly')
import_or_install('bokeh')
import_or_install('matplotlib')
import_or_install('scikit-learn')
import_or_install('streamlit')
import_or_install('hydralit')
import_or_install('teradatasql')
import_or_install('python-dotenv')
import_or_install('teradataml')
import_or_install('streamlit-aggrid')


import os
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import teradatasql
# When importing hydralit you will run into an ModuleNotFoundError: No module named 'streamlit.script_run_context.
# Resolve this error by going into the hydralit code to the sessionstate.py file and adjust codeline 8 as follows:
# R -> File --> open file
# Klik op drie puntjes rechts (naast R logo)
# Kopieer path naar hydralit library. Bij mij is dit: ~/.local/lib/python3.8/site-packages/hydralit
# Open sessionstate.py file, adapt and save as follows:
# CHANGE THIS LINE OF CODE: from streamlit.script_run_context import get_script_run_ctx
# TO THIS LINE OF CODE: from streamlit.scriptrunner import get_script_run_ctx
# see, for more info; https://github.com/TangleSpace/hydralit/issues/27
# NOTE THAT this issue may already be resolved in the future through an open Pull Request on Github
import hydralit as hy
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
from teradataml.dataframe.copy_to import copy_to_sql
from teradataml.context.context import *
# TODO: In the future, we may need to make multiple functions files per tab / overlapping concepts.  
#-----------------------------------------------------------------------------
# DASHBOARD SPECIFIC FUNCTIONS
@st.cache
def processCampaignData(df_campagnes, df_seedlist):
  """
    Process Campaign data for dashboard tables. Certain relevant metrics are computed and these should be 
  """ 
  df_campagnes = df_campagnes.copy()
  #change dtypes
  df_campagnes['Campaign_Type'] = df_campagnes['Campaign_Type'].astype(str)
  df_campagnes['PartnerNaam'] = df_campagnes['PartnerNaam'].astype(str)
  df_campagnes['Sendout_date'] = pd.to_datetime(df_campagnes['Sendout_date'])


  # For filtering purposes in the future it might be useful to extract the months, years and weeknumbers.
  df_campagnes['weeknumber_start'] = df_campagnes['Action_period_startdate'].apply(lambda x:  f"{pd.to_datetime(x).year}_{pd.to_datetime(x).isocalendar()[1]}")
  df_campagnes['weeknumber_end'] = df_campagnes['Action_period_enddate'].apply(lambda x: f"{pd.to_datetime(x).year}_{pd.to_datetime(x).isocalendar()[1]}")
  df_campagnes['year_start'] = df_campagnes['Action_period_startdate'].apply(lambda x : pd.to_datetime(x).year)
  df_campagnes['year_end'] = df_campagnes['Action_period_enddate'].apply(lambda x : pd.to_datetime(x).year)
  df_campagnes['month_start'] = df_campagnes['Action_period_startdate'].apply(lambda x : pd.to_datetime(x).month)
  df_campagnes['month_end'] = df_campagnes['Action_period_enddate'].apply(lambda x : pd.to_datetime(x).month)
  # TO DO: check for Open Rate (OR) if the opens are unique
  # TO DO: check for CLick Through Rate if the clicks are unique
  # TO DO: check for CLick To Open Rate if the clicks and opens are unique
  # TO DO: check if #bounce is hard bounce or soft bounce
  # TO DO: check if optout is the same as unsubscribe to make unsubscribe rate
  
  # rename variables
  df_campagnes.rename(columns = {"total_sent" : "#sent","RESP_ONTVANGEN": "Delivered",
                                  "RESP_OPEN" : "Opens","RESP_CLICK": "#clicks",
                                  "RESP_BOUNCE": "#bounce","RESP_OPTOUT": "#optout",
                                  "RESP_NON_OPEN": "#non_open","n_unique_inw_tx_p": "inw_tx",
                                  "n_unique_sp_tx_p" : "sp_tx","n_unique_don_tx_p":"don_tx",
                                  "total_sendout": "Sent"}
                                  , inplace = True)
                                  
  # add calculated variables
  df_campagnes["%clicks_totalsendout"] = (df_campagnes['#clicks']/df_campagnes['Sent']) * 100
  df_campagnes["%open_totalsendout"] = (df_campagnes['Opens']/df_campagnes['Sent']) * 100
  df_campagnes["%inw_tx_totalsendout"] = (df_campagnes['inw_tx']/df_campagnes['Sent']) * 100
  df_campagnes["%sp_tx_totalsendout"] = (df_campagnes['sp_tx']/df_campagnes['Sent']) * 100
  df_campagnes["%don_tx_totalsendout"] = (df_campagnes['don_tx']/df_campagnes['Sent']) * 100
  df_campagnes["Deliverability Rate (DR)"] = (df_campagnes['Delivered']/df_campagnes['Sent']) * 100
  df_campagnes["Open Rate (OR)"] = (df_campagnes['Opens']/df_campagnes['Delivered']) * 100
  df_campagnes["Click Through Rate (CTR)"] = (df_campagnes['#clicks']/df_campagnes['Delivered']) * 100
  df_campagnes["Click To Open Rate (CTO)"] = (df_campagnes['#clicks']/df_campagnes['Opens']) * 100
  df_campagnes["Bounce Rate (BR)"] = (df_campagnes['#bounce']/df_campagnes['#sent']) * 100
  
  # Join df_campaigns and df_seedlist. 
  df_final = pd.merge(df_campagnes, df_seedlist, on = ['Communication_Name', 'PartnerNaam', 'Campaign_Type', 'Sendout_date'], how = 'left')
  
  # ASSUMPTION: REMOVE ALL CAMPAIGNS THAT WERE SENT TO A SEEDLIST MEMBER AND HAS LESS THAN 20 TOTAL SENDOUTS
  df_final = df_final.loc[~((df_final['contains_seedlist_member'] == 1) & (df_final['Sent'] >= 20))]
  return df_final

def process_campaign_data_tab2(df_campagnes):
  # Count the campaigns that start during a certain week and those that are still ongoing.
  # For now, I assume a two-week period per campaign. Check this code in the future again.
  # TODO: Check how lines of code below may be optimized further.
  campagnes_per_week_start = df_campagnes.groupby(['weeknumber_start', 'Campaign_Type'])['weeknumber_start'].count()
  campagnes_per_week_end = df_campagnes.groupby(['weeknumber_end', 'Campaign_Type'])['weeknumber_end'].count()
  campaigns_per_week = pd.concat([campagnes_per_week_start, campagnes_per_week_end], axis = 1).sum(1).reset_index()
  cols = ['weeknumber', 'Campaign_Type', 'total_campaigns']
  campaigns_per_week.columns = cols
  campaigns_per_week['total_campaigns'] = campaigns_per_week['total_campaigns'].astype(int)
  campaigns_per_week['year_start_campaign'] = campaigns_per_week['weeknumber'].apply(lambda x: int(x[:4]))
  campaigns_per_week['week_number'] = campaigns_per_week['weeknumber'].apply(lambda x: int(x[5:]))
  return campaigns_per_week


@st.cache
# function to get the issuance (uitgifte) or Air Miles in 2021 and 2022
def get_airmiles_uitgifte(host, user, passw):
  query = """SELECT partnernaam, transactiedatum_isojaar, transactiedatum_isoweek, sum(punten) FROM ATP_DBM_VIEW.transactie
    where transactiereden = 'ZSPAREN' AND 
    transactiedatum_isojaar IN (2021, 2022)
    group by 1, 2, 3
    order by 1, 2, 3;"""
  # use with to immediately close connection to db
  with teradatasql.connect(host=host, user=user, password=passw) as connect:
     df = pd.read_sql(query, connect)
  return df

@st.cache
# function to get the redemption (inwisselen) or Air Miles in 2021 and 2022
def get_airmiles_inwisselen(host, user, passw):
  query = """SELECT partnernaam, transactiedatum_isojaar, transactiedatum_isoweek, sum(punten) FROM ATP_DBM_VIEW.transactie
    where transactiereden ('ZINWISSELEN', 'ZAMBONUS') AND 
    transactiedatum_isojaar IN (2021, 2022)
    group by 1, 2, 3
    order by 1, 2, 3;"""
  # use with to immediately close connection to db
  with teradatasql.connect(host=host, user=user, password=passw) as connect:
     df = pd.read_sql(query, connect)
  return df


@st.cache
def get_airmiles_uitgifte_from_db(host, user, passw):
  with teradatasql.connect(host=host, user=user, password=passw) as connect:
    df = pd.read_sql("""SELECT * FROM atp_sandbox.BMO_all_transactions_20212022_aggregate
    WHERE transactiereden IN ('ZSPAREN', 'ZAMBONUS', 'ZBONUS')""", connect)
  return df


# Assumption: For inwisselen KPI, only focus on transactiereden = ZINWISSELEN. 
# This may be changed in the future..
# function to retrieve inwisselen of Air Miles in 2021,2022
# TODO: For now, only focus on ZINWISSELEN. Add ZDONATE later, sometimes the sum(punten) >0 and sometimes < 0 for ZDONATE tx, check why.
@st.cache
def get_airmiles_inwisselen_from_db(host, user, passw):
  with teradatasql.connect(host=host, user=user, password=passw) as connect:
    df = pd.read_sql("""SELECT * FROM atp_sandbox.BMO_all_transactions_20212022_aggregate
    WHERE transactiereden = 'ZINWISSELEN'""", connect)
  return df

# function to access the sanbox table MPA_MainCRMData for retrieving the data needed for the MAIn CRM scatterplot (tab3)
@st.cache
def get_campaign_data_from_db(host, user, passw):
  """
    Extract data for Campaign Dashboard (CRM data). Currently, stored in ATP_SANDBOX table in the future
    this data should be extracted from a Datamart, which already contains all relevant metrics.
  """
  with teradatasql.connect(host=host, user=user, password=passw) as connect:
    df = pd.read_sql("select * from ATP_SANDBOX.MPA_MainCRMData", connect)
  return df

@st.cache
def get_weekly_issuance_budget_from_db(host, user, passw):
  """
   Extract weekly issuance budgets from database.
   These budgets are based on the Excel sheet from the finance department.
  """
  with teradatasql.connect(host=host, user=user, password=passw) as connect:
    # The name is confusing since it gets both budgets for 2021 and 2022.
    # This sandbox table is, however, temporary and will be removed once available in datamart
    df = pd.read_sql("select * from ATP_SANDBOX.bmo_weekly_budget_issuance2021", connect)
  return df

@st.cache
def get_monthly_issuance_budget_from_db(host, user, passw):
  """
   Extract monthly issuance budgets from database.
   Budgets are based on the Excel sheet from the finance department.
  """
  with teradatasql.connect(host=host, user=user, password=passw) as connect:
    # the name is confusing it gets both budgets for 2021 and 2022
    df = pd.read_sql("select * from ATP_SANDBOX.bmo_monthly_budget_issuance2021", connect)
  return df

@st.cache
def get_weekly_redemption_budget_from_db(host, user, passw):
  """
   Extract weekly redemption budgets from database.
   Budgets are based on the Excel sheet from the finance department.
  """
  with teradatasql.connect(host=host, user=user, password=passw) as connect:
    # The name is confusing since it gets both budgets for 2021 and 2022.
    # This sandbox table is, however, temporary and will be removed once available in datamart
    df = pd.read_sql("select * from ATP_SANDBOX.bmo_weekly_budget_redemption2021", connect)
  return df

@st.cache
def get_monthly_redemption_budget_from_db(host, user, passw):
  """
   Extract monthly redemption budgets from database.
   Budgets are based on the Excel sheet from the finance department.
  """
  with teradatasql.connect(host=host, user=user, password=passw) as connect:
    # The name is confusing since it gets both budgets for 2021 and 2022.
    # This sandbox table is, however, temporary and will be removed once available in datamart
    df = pd.read_sql("select * from ATP_SANDBOX.bmo_monthly_budget_redemption2021", connect)
  return df


@st.cache
def extractDataTab3(db_host, db_username, db_password):
  """
    Extracts all data necessary for the Campaign Dashboard (tab 3).
    For now this per this boils down to Campaign Data (ContactHistory table)
  """
  df_campagnes = get_campaign_data_from_db(db_host, db_username, db_password)
  df_seedlist = get_seedlist_campaigns_from_db(db_host, db_username, db_password)
  return df_campagnes, df_seedlist

@st.cache
def processDataTab3(df_campaigns, df_seedlist):
  """
    Data processesing steps for Campaign Dashboard (tab 3).
  """
  df_processed = processCampaignData(df_campaigns, df_seedlist)
  return df_processed

# function to extract all the data for tab 2. 
# Allow_output_mutation = True is needed, since the output is mutated (e.g in the processingTab2 step) (otherwise a warning appears)
@st.cache(allow_output_mutation=True)
def extractDataTab2(db_host, db_username, db_password):
  """
    Extracts all data necessary for KPI Dashboard (tab 2).
    For now this boils down to:
      - Campaign Data (ContactHistory table)
      - Seedlist data (stored in Sandbox table)
      - Transaction data (issuance / redemption)
      - Issuance / Redemption budgets (stored in Sandbox table)
  """
  df_campagnes = get_campaign_data_from_db(db_host, db_username, db_password)
  df_seedlist = get_seedlist_campaigns_from_db(db_host, db_username, db_password)
  df_uitgifte = get_airmiles_uitgifte_from_db(db_host, db_username, db_password)
  df_inwisselingen = get_airmiles_inwisselen_from_db(db_host, db_username, db_password)
  df_issuance_budgets_weekly = get_weekly_issuance_budget_from_db(db_host, db_username, db_password)
  df_issuance_budgets_monthly = get_monthly_issuance_budget_from_db(db_host, db_username, db_password)
  df_redemption_budgets_weekly = get_weekly_redemption_budget_from_db(db_host, db_username, db_password)
  df_redemption_budgets_monthly = get_monthly_redemption_budget_from_db(db_host, db_username, db_password)
  return df_uitgifte, df_inwisselingen, df_campagnes, df_seedlist, df_issuance_budgets_weekly,\
         df_issuance_budgets_monthly, df_redemption_budgets_weekly, df_redemption_budgets_monthly


# function to process the data for tab 2 (extracted by extract)
def processDataTab2(df_uitgifte, df_inwisselingen, df_campagnes, df_seedlist, df_budgets_weekly_issuance, df_budgets_monthly_issuance, df_budgets_weekly_redemption, df_budgets_monthly_redemption):
  """
     Process data for tab2 (KPI Dashboard):
       -  Make inwissel/spaar-partner names across different data sources (e.g. budget and tx data) the same such that filtering
          by user works well on both dataframes.
       -  Add year_isoweek to dataset 
       -  Process Campaigning Data, see function processCampaignData
  """
  # Process uitgifte
  df_uitgifte.loc[:, 'partnernaam'] = df_uitgifte['partnernaam'].str.replace('Praxis Online', 'Praxis')
  df_uitgifte.loc[:, 'partnernaam'] = df_uitgifte['partnernaam'].str.replace('ESSENT', 'Essent')
  df_uitgifte.loc[:, 'partnernaam'] = df_uitgifte['partnernaam'].str.replace('AHonline', 'AH Online')
  df_uitgifte['transactiedatum_year_isoweek'] = df_uitgifte.apply(lambda x: f"{x['transactiedatum_isojaar']}_{x['transactiedatum_isoweek']}", axis = 1)
  # Process Inwisselingen
  df_inwisselingen['punten'] = df_inwisselingen['punten'].abs()
  df_inwisselingen['transactiedatum_year_isoweek'] = df_inwisselingen.apply(lambda x: f"{x['transactiedatum_isojaar']}_{x['transactiedatum_isoweek']}", axis = 1)
  df_inwisselingen['partnernaam'] = df_inwisselingen['partnernaam'].str.replace('Efteling arrangementen', 'Efteling')
  df_inwisselingen['partnernaam'] = df_inwisselingen['partnernaam'].str.replace('Voordeeluitjes.nl', 'Voordeeluitjes')

  # Process Campagnes & seedlist
  df_campagnes = processCampaignData(df_campagnes, df_seedlist)
  # Process budgets weekly
  df_budgets_weekly_issuance['name'] = df_budgets_weekly_issuance['name'].str.replace('Nipo', 'NIPO')
  df_budgets_weekly_issuance['name'] = df_budgets_weekly_issuance['name'].str.replace('AH online', 'AH Online')
  # Process budgets monthly
  df_budgets_monthly_issuance['name'] = df_budgets_monthly_issuance['name'].str.replace('Nipo', 'NIPO')
  df_budgets_monthly_issuance['name'] = df_budgets_monthly_issuance['name'].str.replace('AH online', 'AH Online')
  # Change names such that the name in the budgets dataframe is the same as in df_redemption dataframe
  # This is needed because we need to filter on partner(s) specifically later on through the user input.
  # Ideally, the names would be the same later on in the Datamart.
  df_budgets_weekly_redemption['name'] = df_budgets_weekly_redemption['name'].str.replace('Shell Producten', 'Shell')
  df_budgets_weekly_redemption['name'] = df_budgets_weekly_redemption['name'].str.replace('Shell Brandstof', 'Shell')
  df_budgets_weekly_redemption['name'] = df_budgets_weekly_redemption['name'].str.replace('Beekse Bergen', 'Safaripark Beekse Bergen')
  df_budgets_weekly_redemption['name'] = df_budgets_weekly_redemption['name'].str.replace("Vue Cinema's", 'Vue Cinemas')
  df_budgets_weekly_redemption['name'] = df_budgets_weekly_redemption['name'].str.replace('Dierenpark Amersfoort', 'DierenPark Amersfoort')
  df_budgets_weekly_redemption['name'] = df_budgets_weekly_redemption['name'].str.replace('Shell Brandstof', 'Shell')
  df_budgets_weekly_redemption['name'] = df_budgets_weekly_redemption['name'].str.replace('Pathé evoucher', 'Pathé')
  df_budgets_weekly_redemption['name'] = df_budgets_weekly_redemption['name'].str.replace('Wildlands', 'Wildlands Adventure Zoo Emmen')
  df_budgets_weekly_redemption['name'] = df_budgets_weekly_redemption['name'].str.replace('Producten Tijdelijk', 'Air Miles Producten')
  df_budgets_weekly_redemption['name'] = df_budgets_weekly_redemption['name'].str.replace('Air Miles Shop', 'Air Miles Producten')
  return df_uitgifte, df_inwisselingen, df_campagnes, df_seedlist, df_budgets_weekly_issuance, df_budgets_monthly_issuance

# This function queries on ContactHistoryEvaluatie tabel, which is not efficient. As a temporary fix, we have put it in a sandbox table.
def get_seedlist_campaigns(host, user, passw):
  """
    Testing campaigns are send to memberid's that are on the seedlist (marketing or another team) uses these
    To test whether the content of the email is fine and every link is working properly.
    For an overview of the seedlist memberid's go to Teams --> Team MarCom --> LMN Seedlist
  
    TODO: In the future, we may want to add this as well to a sandbox table rather than querying directly on the db
    Make sure that the query is the same as the one in get_campaign_data() as we are joining both df's later
  """

  query_seedlist_campaigns = """ SELECT Communication_Name, PartnerNaam, Campaign_Type, Sendout_Date, action_period_startdate, action_period_enddate FROM ATP_DBM_VIEW.ContactHistoryEvaluatie
  WHERE Sendout_Date >= '2021-01-01'
  AND membershipid in (3135803, 2221636, 5572048, 7030800,4349458,5713801,1989732,6077775,6992929,7030478,3820427,483645,3410843,6947311,6282933,3117486,7245151)
  group by 1, 2, 3, 4, 5, 6;"""

  with teradatasql.connect(host=host, user=user, password=passw) as connect:
    df_seedlist = pd.read_sql(query_seedlist_campaigns, connect)
  
  # Convert columns to right type before joining (otherwise the joining fails later on in the processing step)
  # Also add a column that will later be used to distinguish between seedlisted campaigns and those that are not.
  df_seedlist['Campaign_Type'] = df_seedlist['Campaign_Type'].astype(str)
  df_seedlist['PartnerNaam'] = df_seedlist['PartnerNaam'].astype(str)
  df_seedlist['Sendout_date'] = pd.to_datetime(df_seedlist['Sendout_date'])
  df_seedlist['contains_seedlist_member'] = 1
  return df_seedlist

# Function to access the sandbox table BMO_seedlist_campaigns for retrieving the seedlist
@st.cache
def get_seedlist_campaigns_from_db(host, user, passw):
  """
    Testing campaigns are send to memberid's that are on the seedlist (marketing or another team) uses these
    To test whether the content of the email is fine and every link is working properly.
    For an overview of the seedlist memberid's go to Teams --> Team MarCom --> LMN Seedlist
    
    TODO: Should be scheduled in the future.
  """

  query_seedlist_campaigns = """ SELECT * FROM ATP_SANDBOX.BMO_seedlist_campaigns;"""

  with teradatasql.connect(host=host, user=user, password=passw) as connect:
    df_seedlist = pd.read_sql(query_seedlist_campaigns, connect)
  return df_seedlist


def write_seedlist_to_db(host, user, passw):
  """
    Test this function, doesn't work yet. Also, in the future the query from the get_seedlist_campaigns function
    needs to be scheduled.
  """
  df_seedlist = get_seedlist_campaigns(host, user, passw)
  con = create_context(host = host, user = user, password = passw) 
  try:
    copy_to_sql(df = df_seedlist, table_name = 'BMO_seedlist_campaigns',
                schema_name = 'ATP_SANDBOX', if_exists = 'replace')
    print("Successfully added the table to DB")
  except Exception as e:
    print(f"Exception {e}. Not Successful, please check")
    
    
  
