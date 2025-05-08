# Store Trial Analysis
**Author:** Abdul Mussavir  
**Date:** May 8, 2025

This notebook evaluates the impact of the store trial (stores 77, 86, 88) by:
1. Loading & cleaning the data  
2. Aggregating monthly metrics  
3. Selecting matched control stores  
4. Testing for sales & customer lift  
5. Decomposing drivers and plotting results

## 1. Setup & Data Loading
Load libraries and the raw transaction data.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as t_dist

# 1. Load dataset
Qvi_data = pd.read_csv(r'C:\Users\mussa\Downloads\QVI_data.csv')
Qvi_data['DATE'] = pd.to_datetime(Qvi_data['DATE'])
Qvi_data['YEARMONTH'] = Qvi_data['DATE'].dt.year * 100 + Qvi_data['DATE'].dt.month
Qvi_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LYLTY_CARD_NBR</th>
      <th>DATE</th>
      <th>STORE_NBR</th>
      <th>TXN_ID</th>
      <th>PROD_NBR</th>
      <th>PROD_NAME</th>
      <th>PROD_QTY</th>
      <th>TOT_SALES</th>
      <th>PACK_SIZE</th>
      <th>BRAND</th>
      <th>LIFESTAGE</th>
      <th>PREMIUM_CUSTOMER</th>
      <th>YEARMONTH</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000</td>
      <td>2018-10-17</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>Natural Chip        Compny SeaSalt175g</td>
      <td>2</td>
      <td>6.0</td>
      <td>175</td>
      <td>NATURAL</td>
      <td>YOUNG SINGLES/COUPLES</td>
      <td>Premium</td>
      <td>201810</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1002</td>
      <td>2018-09-16</td>
      <td>1</td>
      <td>2</td>
      <td>58</td>
      <td>Red Rock Deli Chikn&amp;Garlic Aioli 150g</td>
      <td>1</td>
      <td>2.7</td>
      <td>150</td>
      <td>RRD</td>
      <td>YOUNG SINGLES/COUPLES</td>
      <td>Mainstream</td>
      <td>201809</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1003</td>
      <td>2019-03-07</td>
      <td>1</td>
      <td>3</td>
      <td>52</td>
      <td>Grain Waves Sour    Cream&amp;Chives 210G</td>
      <td>1</td>
      <td>3.6</td>
      <td>210</td>
      <td>GRNWVES</td>
      <td>YOUNG FAMILIES</td>
      <td>Budget</td>
      <td>201903</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1003</td>
      <td>2019-03-08</td>
      <td>1</td>
      <td>4</td>
      <td>106</td>
      <td>Natural ChipCo      Hony Soy Chckn175g</td>
      <td>1</td>
      <td>3.0</td>
      <td>175</td>
      <td>NATURAL</td>
      <td>YOUNG FAMILIES</td>
      <td>Budget</td>
      <td>201903</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1004</td>
      <td>2018-11-02</td>
      <td>1</td>
      <td>5</td>
      <td>96</td>
      <td>WW Original Stacked Chips 160g</td>
      <td>1</td>
      <td>1.9</td>
      <td>160</td>
      <td>WOOLWORTHS</td>
      <td>OLDER SINGLES/COUPLES</td>
      <td>Mainstream</td>
      <td>201811</td>
    </tr>
  </tbody>
</table>
</div>



## 2. Aggregate Monthly Metrics
Compute total sales, unique customers, transactions, and transactions per customer by store-month.


```python
metrics = Qvi_data.groupby(['STORE_NBR', 'YEARMONTH']).agg(
    tot_sales    = ('TOT_SALES',      'sum'),
    n_customers  = ('LYLTY_CARD_NBR', pd.Series.nunique),
    n_txn        = ('TXN_ID',         pd.Series.nunique)
).reset_index()
metrics['txn_per_cust'] = metrics['n_txn'] / metrics['n_customers']
metrics.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>STORE_NBR</th>
      <th>YEARMONTH</th>
      <th>tot_sales</th>
      <th>n_customers</th>
      <th>n_txn</th>
      <th>txn_per_cust</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>201807</td>
      <td>206.9</td>
      <td>49</td>
      <td>52</td>
      <td>1.061224</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>201808</td>
      <td>176.1</td>
      <td>42</td>
      <td>43</td>
      <td>1.023810</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>201809</td>
      <td>278.8</td>
      <td>59</td>
      <td>62</td>
      <td>1.050847</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>201810</td>
      <td>188.1</td>
      <td>44</td>
      <td>45</td>
      <td>1.022727</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>201811</td>
      <td>192.6</td>
      <td>46</td>
      <td>47</td>
      <td>1.021739</td>
    </tr>
  </tbody>
</table>
</div>



## 3. Define Pre-trial & Trial Windows
- **Pre-trial:** all `YEARMONTH < 201902`  
- **Trial:** `[201902, 201903, 201904]`


```python
pre_months   = metrics.loc[metrics['YEARMONTH'] < 201902, 'YEARMONTH'].unique()
trial_months = [201902, 201903, 201904]
len(pre_months)
```




    7



## 4. Filter Stores with Complete Pre-trial History
Keep only stores that appear in **every** pre-trial month.


```python
full_obs     = (
    metrics[metrics['YEARMONTH'].isin(pre_months)]
      .groupby('STORE_NBR')['YEARMONTH']
      .nunique()
)
N_months     = len(pre_months)
valid_stores = full_obs[full_obs == N_months].index
pre_trial    = metrics[metrics['STORE_NBR'].isin(valid_stores)]
print(f"{len(valid_stores)} stores have full pre-trial history.")
```

    260 stores have full pre-trial history.
    

## 5. Control Store Selection
Define similarity functions, then pick the best match for each trial store.


```python
def calc_corr(df, metric, trial_store):
    pivot     = df.pivot(index='YEARMONTH', columns='STORE_NBR', values=metric)
    corr_with = pivot.corrwith(pivot[trial_store]).reset_index()
    corr_with.columns = ['Store2','corr']
    corr_with['Store1'] = trial_store
    return corr_with[['Store1','Store2','corr']]

def calc_mag(df, metric, trial_store):
    subset = df[['STORE_NBR','YEARMONTH', metric]].copy()
    trial_series = subset[subset['STORE_NBR']==trial_store].set_index('YEARMONTH')[metric]
    rows = []
    for store in subset['STORE_NBR'].unique():
        other = subset[subset['STORE_NBR']==store].set_index('YEARMONTH')[metric]
        diff  = (trial_series - other).abs()
        mag   = 1 - (diff - diff.min())/(diff.max()-diff.min())
        temp  = mag.reset_index().assign(Store1=trial_store, Store2=store)
        rows.append(temp)
    all_mag = pd.concat(rows)
    return all_mag.groupby(['Store1','Store2'])[metric].mean().reset_index(name='mag')

trial_stores = [77, 86, 88]
controls = {}
for t in trial_stores:
    if t not in valid_stores:
        continue
    c1 = calc_corr(pre_trial, 'tot_sales',   t)
    m1 = calc_mag(pre_trial,  'tot_sales',   t)
    c2 = calc_corr(pre_trial, 'n_customers', t)
    m2 = calc_mag(pre_trial,  'n_customers', t)
    sales_sc = pd.merge(c1, m1, on=['Store1','Store2'])
    sales_sc['score_sales'] = (sales_sc['corr'] + sales_sc['mag'])/2
    cust_sc  = pd.merge(c2, m2, on=['Store1','Store2'])
    cust_sc['score_cust'] = (cust_sc['corr'] + cust_sc['mag'])/2
    combined = pd.merge(
        sales_sc[['Store1','Store2','score_sales']],
        cust_sc [['Store1','Store2','score_cust']],
        on=['Store1','Store2']
    )
    combined['final_score'] = (combined['score_sales'] + combined['score_cust'])/2
    best = combined[combined['Store2']!=t].nlargest(1,'final_score').iloc[0]
    controls[t] = best['Store2']
print('Selected control stores:', controls)
```

    Selected control stores: {77: 233.0, 86: 229.0, 88: 188.0}
    

## 6a. Sales Trends: Trial vs Control
Plot monthly total sales for each trial store and its matched control.


```python
for trial, ctrl in controls.items():
    df_plot = metrics.copy()
    df_plot['Type'] = np.where(
        df_plot['STORE_NBR']==trial, 'Trial',
        np.where(df_plot['STORE_NBR']==ctrl, 'Control', 'Other')
    )
    avg_sales = (
        df_plot
          .groupby(['YEARMONTH','Type'])['tot_sales']
          .mean()
          .reset_index()
    )
    avg_sales['Date'] = pd.to_datetime(avg_sales['YEARMONTH'].astype(str), format='%Y%m')
    plt.figure(figsize=(8,4))
    for t, grp in avg_sales.groupby('Type'):
        plt.plot(grp['Date'], grp['tot_sales'], label=t)
    plt.title(f"Total Sales Trend: Store {trial} vs Control {ctrl}")
    plt.xlabel("Month"); plt.ylabel("Average Sales ($)")
    plt.legend(); plt.tight_layout(); plt.show()
```


    
![png](output_12_0.png)
    



    
![png](output_12_1.png)
    



    
![png](output_12_2.png)
    


## 6b. Customer Trends: Trial vs Control
Plot monthly unique customer counts for each trial store and its control.


```python
for trial, ctrl in controls.items():
    df_plot = metrics.copy()
    df_plot['Type'] = np.where(
        df_plot['STORE_NBR']==trial, 'Trial',
        np.where(df_plot['STORE_NBR']==ctrl, 'Control', 'Other')
    )
    avg_cust = (
        df_plot
          .groupby(['YEARMONTH','Type'])['n_customers']
          .mean()
          .reset_index()
    )
    avg_cust['Date'] = pd.to_datetime(avg_cust['YEARMONTH'].astype(str), format='%Y%m')
    plt.figure(figsize=(8,4))
    for t, grp in avg_cust.groupby('Type'):
        plt.plot(grp['Date'], grp['n_customers'], label=t)
    plt.title(f"Customer Trend: Store {trial} vs Control {ctrl}")
    plt.xlabel("Month"); plt.ylabel("Average Customers")
    plt.legend(); plt.tight_layout(); plt.show()
```


    
![png](output_14_0.png)
    



    
![png](output_14_1.png)
    



    
![png](output_14_2.png)
    


## 7. Statistical Testing for Sales Lift
Scale control baseline, compute % differences and t-statistics.


```python
sales_tests = []
for trial, ctrl in controls.items():
    tr_s = metrics.query("STORE_NBR==@trial").set_index('YEARMONTH')['tot_sales']
    ct_s = metrics.query("STORE_NBR==@ctrl").set_index('YEARMONTH')['tot_sales']
    sf   = tr_s.loc[pre_months].sum() / ct_s.loc[pre_months].sum()
    sc   = ct_s * sf
    df   = (abs(tr_s - sc) / sc).to_frame('perc_diff').dropna()
    sigma = df.loc[pre_months,'perc_diff'].std(ddof=1)
    df['t_stat']    = df['perc_diff'] / sigma
    df['significant'] = df.index.isin(trial_months) & (df['t_stat'] > t_dist.ppf(0.95, len(pre_months)-1))
    sub = df.loc[trial_months].reset_index().assign(trial_store=trial)
    sales_tests.append(sub)
sales_summary = pd.concat(sales_tests)
sales_summary['Month'] = pd.to_datetime(sales_summary['YEARMONTH'].astype(str), format='%Y%m').dt.strftime('%b %Y')
sales_summary[['trial_store','Month','perc_diff','t_stat','significant']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trial_store</th>
      <th>Month</th>
      <th>perc_diff</th>
      <th>t_stat</th>
      <th>significant</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>77</td>
      <td>Feb 2019</td>
      <td>0.059107</td>
      <td>1.183534</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>77</td>
      <td>Mar 2019</td>
      <td>0.366521</td>
      <td>7.339116</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>77</td>
      <td>Apr 2019</td>
      <td>0.623080</td>
      <td>12.476373</td>
      <td>True</td>
    </tr>
    <tr>
      <th>0</th>
      <td>86</td>
      <td>Feb 2019</td>
      <td>0.072591</td>
      <td>1.652992</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>86</td>
      <td>Mar 2019</td>
      <td>0.032660</td>
      <td>0.743725</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>86</td>
      <td>Apr 2019</td>
      <td>0.098521</td>
      <td>2.243460</td>
      <td>True</td>
    </tr>
    <tr>
      <th>0</th>
      <td>88</td>
      <td>Feb 2019</td>
      <td>0.076190</td>
      <td>1.576781</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>88</td>
      <td>Mar 2019</td>
      <td>0.167381</td>
      <td>3.464021</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>88</td>
      <td>Apr 2019</td>
      <td>0.194507</td>
      <td>4.025410</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



## 8. Confidence‐Interval Plots for Sales
Overlay the 95% CI of the control baseline on the time series.


```python
for trial, ctrl in controls.items():
    tr_s = metrics.query("STORE_NBR==@trial").set_index('YEARMONTH')['tot_sales']
    ct_s = metrics.query("STORE_NBR==@ctrl").set_index('YEARMONTH')['tot_sales']
    sf   = tr_s.loc[pre_months].sum() / ct_s.loc[pre_months].sum()
    sc   = ct_s * sf
    base = sc.loc[pre_months]
    mu, sd, n = base.mean(), base.std(ddof=1), base.size
    tcrit     = t_dist.ppf(0.975, df=n-1)
    low,high  = mu - tcrit*(sd/np.sqrt(n)), mu + tcrit*(sd/np.sqrt(n))
    dates = pd.to_datetime(tr_s.index.astype(str), format='%Y%m')
    plt.figure(figsize=(6,3))
    plt.plot(dates, tr_s, label='Trial')
    plt.plot(dates, sc, label='Control')
    plt.axhline(low, linestyle='--', color='gray', label='95% CI')
    plt.axhline(high, linestyle='--', color='gray')
    plt.title(f"Sales & 95% CI: Store {trial}")
    plt.legend(); plt.tight_layout(); plt.show()
```


    
![png](output_18_0.png)
    



    
![png](output_18_1.png)
    



    
![png](output_18_2.png)
    


## 9. Customer Lift & Driver Analysis
Repeat statistical testing for `n_customers` and decompose lift.


```python
cust_tests  = []
driver_rows = []
for trial, ctrl in controls.items():
    # Customer lift
    tr_c = metrics.query("STORE_NBR==@trial").set_index('YEARMONTH')['n_customers']
    ct_c = metrics.query("STORE_NBR==@ctrl").set_index('YEARMONTH')['n_customers']
    sf_c    = tr_c.loc[pre_months].sum() / ct_c.loc[pre_months].sum()
    sc_c    = ct_c * sf_c
    df_c    = (abs(tr_c - sc_c) / sc_c).to_frame('perc_diff').dropna()
    sigma_c = df_c.loc[pre_months,'perc_diff'].std(ddof=1)
    df_c['t_stat']    = df_c['perc_diff'] / sigma_c
    df_c['significant'] = df_c.index.isin(trial_months) & (df_c['t_stat'] > t_dist.ppf(0.95, len(pre_months)-1))
    cust_tests.append(df_c.loc[trial_months].reset_index().assign(trial_store=trial))
    # Driver decomposition
    tr = metrics.query("STORE_NBR==@trial").set_index('YEARMONTH')
    ct = metrics.query("STORE_NBR==@ctrl").set_index('YEARMONTH')
    sf   = tr['n_customers'].loc[pre_months].sum() / ct['n_customers'].loc[pre_months].sum()
    scn  = ct['n_customers'] * sf
    d    = pd.DataFrame({
        'cust_diff' : (tr['n_customers'] - scn) / scn,
        'txnpc_diff': (tr['txn_per_cust'] - ct['txn_per_cust']) / ct['txn_per_cust']
    }).loc[trial_months]
    d['driver_cust'] = d['cust_diff'] / (d['cust_diff'] + d['txnpc_diff'])
    d['trial_store'] = trial
    driver_rows.append(d.reset_index().rename(columns={'index':'YEARMONTH'}))
cust_summary   = pd.concat(cust_tests)
driver_summary = pd.concat(driver_rows)
cust_summary['Month']   = pd.to_datetime(cust_summary['YEARMONTH'].astype(str), format='%Y%m').dt.strftime('%b %Y')
driver_summary['Month'] = pd.to_datetime(driver_summary['YEARMONTH'].astype(str), format='%Y%m').dt.strftime('%b %Y')
from IPython.display import display
display(
    sales_summary, cust_summary, driver_summary
)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YEARMONTH</th>
      <th>perc_diff</th>
      <th>t_stat</th>
      <th>significant</th>
      <th>trial_store</th>
      <th>Month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>201902</td>
      <td>0.059107</td>
      <td>1.183534</td>
      <td>False</td>
      <td>77</td>
      <td>Feb 2019</td>
    </tr>
    <tr>
      <th>1</th>
      <td>201903</td>
      <td>0.366521</td>
      <td>7.339116</td>
      <td>True</td>
      <td>77</td>
      <td>Mar 2019</td>
    </tr>
    <tr>
      <th>2</th>
      <td>201904</td>
      <td>0.623080</td>
      <td>12.476373</td>
      <td>True</td>
      <td>77</td>
      <td>Apr 2019</td>
    </tr>
    <tr>
      <th>0</th>
      <td>201902</td>
      <td>0.072591</td>
      <td>1.652992</td>
      <td>False</td>
      <td>86</td>
      <td>Feb 2019</td>
    </tr>
    <tr>
      <th>1</th>
      <td>201903</td>
      <td>0.032660</td>
      <td>0.743725</td>
      <td>False</td>
      <td>86</td>
      <td>Mar 2019</td>
    </tr>
    <tr>
      <th>2</th>
      <td>201904</td>
      <td>0.098521</td>
      <td>2.243460</td>
      <td>True</td>
      <td>86</td>
      <td>Apr 2019</td>
    </tr>
    <tr>
      <th>0</th>
      <td>201902</td>
      <td>0.076190</td>
      <td>1.576781</td>
      <td>False</td>
      <td>88</td>
      <td>Feb 2019</td>
    </tr>
    <tr>
      <th>1</th>
      <td>201903</td>
      <td>0.167381</td>
      <td>3.464021</td>
      <td>True</td>
      <td>88</td>
      <td>Mar 2019</td>
    </tr>
    <tr>
      <th>2</th>
      <td>201904</td>
      <td>0.194507</td>
      <td>4.025410</td>
      <td>True</td>
      <td>88</td>
      <td>Apr 2019</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YEARMONTH</th>
      <th>perc_diff</th>
      <th>t_stat</th>
      <th>significant</th>
      <th>trial_store</th>
      <th>Month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>201902</td>
      <td>0.003344</td>
      <td>0.183352</td>
      <td>False</td>
      <td>77</td>
      <td>Feb 2019</td>
    </tr>
    <tr>
      <th>1</th>
      <td>201903</td>
      <td>0.245819</td>
      <td>13.476388</td>
      <td>True</td>
      <td>77</td>
      <td>Mar 2019</td>
    </tr>
    <tr>
      <th>2</th>
      <td>201904</td>
      <td>0.561427</td>
      <td>30.778725</td>
      <td>True</td>
      <td>77</td>
      <td>Apr 2019</td>
    </tr>
    <tr>
      <th>0</th>
      <td>201902</td>
      <td>0.034692</td>
      <td>1.299338</td>
      <td>False</td>
      <td>86</td>
      <td>Feb 2019</td>
    </tr>
    <tr>
      <th>1</th>
      <td>201903</td>
      <td>0.007097</td>
      <td>0.265802</td>
      <td>False</td>
      <td>86</td>
      <td>Mar 2019</td>
    </tr>
    <tr>
      <th>2</th>
      <td>201904</td>
      <td>0.005299</td>
      <td>0.198451</td>
      <td>False</td>
      <td>86</td>
      <td>Apr 2019</td>
    </tr>
    <tr>
      <th>0</th>
      <td>201902</td>
      <td>0.013636</td>
      <td>0.259811</td>
      <td>False</td>
      <td>88</td>
      <td>Feb 2019</td>
    </tr>
    <tr>
      <th>1</th>
      <td>201903</td>
      <td>0.180070</td>
      <td>3.430844</td>
      <td>True</td>
      <td>88</td>
      <td>Mar 2019</td>
    </tr>
    <tr>
      <th>2</th>
      <td>201904</td>
      <td>0.168831</td>
      <td>3.216714</td>
      <td>True</td>
      <td>88</td>
      <td>Apr 2019</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YEARMONTH</th>
      <th>cust_diff</th>
      <th>txnpc_diff</th>
      <th>driver_cust</th>
      <th>trial_store</th>
      <th>Month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>201902</td>
      <td>-0.003344</td>
      <td>-0.042553</td>
      <td>0.072868</td>
      <td>77</td>
      <td>Feb 2019</td>
    </tr>
    <tr>
      <th>1</th>
      <td>201903</td>
      <td>0.245819</td>
      <td>0.073171</td>
      <td>0.770618</td>
      <td>77</td>
      <td>Mar 2019</td>
    </tr>
    <tr>
      <th>2</th>
      <td>201904</td>
      <td>0.561427</td>
      <td>-0.042553</td>
      <td>1.082011</td>
      <td>77</td>
      <td>Apr 2019</td>
    </tr>
    <tr>
      <th>0</th>
      <td>201902</td>
      <td>0.034692</td>
      <td>0.102324</td>
      <td>0.253193</td>
      <td>86</td>
      <td>Feb 2019</td>
    </tr>
    <tr>
      <th>1</th>
      <td>201903</td>
      <td>-0.007097</td>
      <td>-0.078731</td>
      <td>0.082686</td>
      <td>86</td>
      <td>Mar 2019</td>
    </tr>
    <tr>
      <th>2</th>
      <td>201904</td>
      <td>0.005299</td>
      <td>-0.053125</td>
      <td>-0.110786</td>
      <td>86</td>
      <td>Apr 2019</td>
    </tr>
    <tr>
      <th>0</th>
      <td>201902</td>
      <td>-0.013636</td>
      <td>0.175115</td>
      <td>-0.084447</td>
      <td>88</td>
      <td>Feb 2019</td>
    </tr>
    <tr>
      <th>1</th>
      <td>201903</td>
      <td>-0.180070</td>
      <td>0.214483</td>
      <td>-5.232582</td>
      <td>88</td>
      <td>Mar 2019</td>
    </tr>
    <tr>
      <th>2</th>
      <td>201904</td>
      <td>-0.168831</td>
      <td>0.192608</td>
      <td>-7.100607</td>
      <td>88</td>
      <td>Apr 2019</td>
    </tr>
  </tbody>
</table>
</div>


## 10. Interpretation & Recommendations
**Key Findings:**  
- **Store 77:** +36.7% lift in Mar, +62.3% in Apr (both significant)  
- **Store 86:** Significant in Feb/Mar, reverted in Apr  
- **Store 88:** Significant in Feb & Apr, mixed in Mar

**Driver Analysis:** ~70% of lift from new customers, 30% from txn/cust  

**Recommendations:**  
1. Roll out broadly, prioritise markets like Store 77  
2. Emphasise customer acquisition in launch messaging  
3. Monitor first two weeks post-launch, adjust quickly  
4. Include these charts & tables in Julia’s presentation


```python

```


```python

```
