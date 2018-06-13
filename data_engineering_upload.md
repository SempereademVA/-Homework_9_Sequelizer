

```python
%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import datetime as dt
```


```python
########################################################################################################
```


```python
raw_measurements_df = pd.read_csv('hawaii_measurements.csv')
raw_stations_df = pd.read_csv('hawaii_stations.csv')
```


```python
raw_measurements_df.tail()
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
      <th>station</th>
      <th>date</th>
      <th>prcp</th>
      <th>tobs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19545</th>
      <td>USC00516128</td>
      <td>2017-08-19</td>
      <td>0.09</td>
      <td>71</td>
    </tr>
    <tr>
      <th>19546</th>
      <td>USC00516128</td>
      <td>2017-08-20</td>
      <td>NaN</td>
      <td>78</td>
    </tr>
    <tr>
      <th>19547</th>
      <td>USC00516128</td>
      <td>2017-08-21</td>
      <td>0.56</td>
      <td>76</td>
    </tr>
    <tr>
      <th>19548</th>
      <td>USC00516128</td>
      <td>2017-08-22</td>
      <td>0.50</td>
      <td>76</td>
    </tr>
    <tr>
      <th>19549</th>
      <td>USC00516128</td>
      <td>2017-08-23</td>
      <td>0.45</td>
      <td>76</td>
    </tr>
  </tbody>
</table>
</div>




```python
raw_measurements_df.dtypes
```




    station     object
    date        object
    prcp       float64
    tobs         int64
    dtype: object




```python
#Find duplicate rows
search = pd.DataFrame.duplicated(raw_measurements_df)
search[search == True]
```




    Series([], dtype: bool)



No duplicate rows found

Look for missing data


```python
raw_measurements_df.shape
```




    (19550, 4)




```python
raw_measurements_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 19550 entries, 0 to 19549
    Data columns (total 4 columns):
    station    19550 non-null object
    date       19550 non-null object
    prcp       18103 non-null float64
    tobs       19550 non-null int64
    dtypes: float64(1), int64(1), object(2)
    memory usage: 611.0+ KB
    


```python
raw_measurements_df.describe()
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
      <th>prcp</th>
      <th>tobs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>18103.000000</td>
      <td>19550.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.160644</td>
      <td>73.097954</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.468746</td>
      <td>4.523527</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>53.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>70.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.010000</td>
      <td>73.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.110000</td>
      <td>76.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>11.530000</td>
      <td>87.000000</td>
    </tr>
  </tbody>
</table>
</div>



There is data missing in the raw measurements dataframe. prcp has only 18103 values 


```python
#Drop rows with NaN or null values
clean_raw_measurements_df= raw_measurements_df.dropna()
clean_raw_measurements_df
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
      <th>station</th>
      <th>date</th>
      <th>prcp</th>
      <th>tobs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>USC00519397</td>
      <td>2010-01-01</td>
      <td>0.08</td>
      <td>65</td>
    </tr>
    <tr>
      <th>1</th>
      <td>USC00519397</td>
      <td>2010-01-02</td>
      <td>0.00</td>
      <td>63</td>
    </tr>
    <tr>
      <th>2</th>
      <td>USC00519397</td>
      <td>2010-01-03</td>
      <td>0.00</td>
      <td>74</td>
    </tr>
    <tr>
      <th>3</th>
      <td>USC00519397</td>
      <td>2010-01-04</td>
      <td>0.00</td>
      <td>76</td>
    </tr>
    <tr>
      <th>5</th>
      <td>USC00519397</td>
      <td>2010-01-07</td>
      <td>0.06</td>
      <td>70</td>
    </tr>
    <tr>
      <th>6</th>
      <td>USC00519397</td>
      <td>2010-01-08</td>
      <td>0.00</td>
      <td>64</td>
    </tr>
    <tr>
      <th>7</th>
      <td>USC00519397</td>
      <td>2010-01-09</td>
      <td>0.00</td>
      <td>68</td>
    </tr>
    <tr>
      <th>8</th>
      <td>USC00519397</td>
      <td>2010-01-10</td>
      <td>0.00</td>
      <td>73</td>
    </tr>
    <tr>
      <th>9</th>
      <td>USC00519397</td>
      <td>2010-01-11</td>
      <td>0.01</td>
      <td>64</td>
    </tr>
    <tr>
      <th>10</th>
      <td>USC00519397</td>
      <td>2010-01-12</td>
      <td>0.00</td>
      <td>61</td>
    </tr>
    <tr>
      <th>11</th>
      <td>USC00519397</td>
      <td>2010-01-14</td>
      <td>0.00</td>
      <td>66</td>
    </tr>
    <tr>
      <th>12</th>
      <td>USC00519397</td>
      <td>2010-01-15</td>
      <td>0.00</td>
      <td>65</td>
    </tr>
    <tr>
      <th>13</th>
      <td>USC00519397</td>
      <td>2010-01-16</td>
      <td>0.00</td>
      <td>68</td>
    </tr>
    <tr>
      <th>14</th>
      <td>USC00519397</td>
      <td>2010-01-17</td>
      <td>0.00</td>
      <td>64</td>
    </tr>
    <tr>
      <th>15</th>
      <td>USC00519397</td>
      <td>2010-01-18</td>
      <td>0.00</td>
      <td>72</td>
    </tr>
    <tr>
      <th>16</th>
      <td>USC00519397</td>
      <td>2010-01-19</td>
      <td>0.00</td>
      <td>66</td>
    </tr>
    <tr>
      <th>17</th>
      <td>USC00519397</td>
      <td>2010-01-20</td>
      <td>0.00</td>
      <td>66</td>
    </tr>
    <tr>
      <th>18</th>
      <td>USC00519397</td>
      <td>2010-01-21</td>
      <td>0.00</td>
      <td>69</td>
    </tr>
    <tr>
      <th>19</th>
      <td>USC00519397</td>
      <td>2010-01-22</td>
      <td>0.00</td>
      <td>67</td>
    </tr>
    <tr>
      <th>20</th>
      <td>USC00519397</td>
      <td>2010-01-23</td>
      <td>0.00</td>
      <td>67</td>
    </tr>
    <tr>
      <th>21</th>
      <td>USC00519397</td>
      <td>2010-01-24</td>
      <td>0.01</td>
      <td>71</td>
    </tr>
    <tr>
      <th>22</th>
      <td>USC00519397</td>
      <td>2010-01-25</td>
      <td>0.00</td>
      <td>67</td>
    </tr>
    <tr>
      <th>23</th>
      <td>USC00519397</td>
      <td>2010-01-26</td>
      <td>0.04</td>
      <td>76</td>
    </tr>
    <tr>
      <th>24</th>
      <td>USC00519397</td>
      <td>2010-01-27</td>
      <td>0.12</td>
      <td>68</td>
    </tr>
    <tr>
      <th>25</th>
      <td>USC00519397</td>
      <td>2010-01-28</td>
      <td>0.00</td>
      <td>72</td>
    </tr>
    <tr>
      <th>27</th>
      <td>USC00519397</td>
      <td>2010-01-31</td>
      <td>0.03</td>
      <td>67</td>
    </tr>
    <tr>
      <th>28</th>
      <td>USC00519397</td>
      <td>2010-02-01</td>
      <td>0.01</td>
      <td>66</td>
    </tr>
    <tr>
      <th>30</th>
      <td>USC00519397</td>
      <td>2010-02-04</td>
      <td>0.01</td>
      <td>69</td>
    </tr>
    <tr>
      <th>31</th>
      <td>USC00519397</td>
      <td>2010-02-05</td>
      <td>0.00</td>
      <td>67</td>
    </tr>
    <tr>
      <th>32</th>
      <td>USC00519397</td>
      <td>2010-02-06</td>
      <td>0.00</td>
      <td>67</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>19513</th>
      <td>USC00516128</td>
      <td>2017-07-17</td>
      <td>0.39</td>
      <td>72</td>
    </tr>
    <tr>
      <th>19514</th>
      <td>USC00516128</td>
      <td>2017-07-18</td>
      <td>2.40</td>
      <td>77</td>
    </tr>
    <tr>
      <th>19515</th>
      <td>USC00516128</td>
      <td>2017-07-19</td>
      <td>0.27</td>
      <td>74</td>
    </tr>
    <tr>
      <th>19516</th>
      <td>USC00516128</td>
      <td>2017-07-20</td>
      <td>0.70</td>
      <td>75</td>
    </tr>
    <tr>
      <th>19517</th>
      <td>USC00516128</td>
      <td>2017-07-21</td>
      <td>0.10</td>
      <td>72</td>
    </tr>
    <tr>
      <th>19518</th>
      <td>USC00516128</td>
      <td>2017-07-22</td>
      <td>4.00</td>
      <td>72</td>
    </tr>
    <tr>
      <th>19519</th>
      <td>USC00516128</td>
      <td>2017-07-23</td>
      <td>0.80</td>
      <td>78</td>
    </tr>
    <tr>
      <th>19520</th>
      <td>USC00516128</td>
      <td>2017-07-24</td>
      <td>0.84</td>
      <td>77</td>
    </tr>
    <tr>
      <th>19521</th>
      <td>USC00516128</td>
      <td>2017-07-25</td>
      <td>0.30</td>
      <td>79</td>
    </tr>
    <tr>
      <th>19522</th>
      <td>USC00516128</td>
      <td>2017-07-26</td>
      <td>0.30</td>
      <td>73</td>
    </tr>
    <tr>
      <th>19523</th>
      <td>USC00516128</td>
      <td>2017-07-27</td>
      <td>0.00</td>
      <td>75</td>
    </tr>
    <tr>
      <th>19524</th>
      <td>USC00516128</td>
      <td>2017-07-28</td>
      <td>0.40</td>
      <td>73</td>
    </tr>
    <tr>
      <th>19525</th>
      <td>USC00516128</td>
      <td>2017-07-29</td>
      <td>0.30</td>
      <td>77</td>
    </tr>
    <tr>
      <th>19526</th>
      <td>USC00516128</td>
      <td>2017-07-30</td>
      <td>0.30</td>
      <td>79</td>
    </tr>
    <tr>
      <th>19527</th>
      <td>USC00516128</td>
      <td>2017-07-31</td>
      <td>0.00</td>
      <td>74</td>
    </tr>
    <tr>
      <th>19529</th>
      <td>USC00516128</td>
      <td>2017-08-02</td>
      <td>0.25</td>
      <td>80</td>
    </tr>
    <tr>
      <th>19530</th>
      <td>USC00516128</td>
      <td>2017-08-03</td>
      <td>0.06</td>
      <td>76</td>
    </tr>
    <tr>
      <th>19533</th>
      <td>USC00516128</td>
      <td>2017-08-07</td>
      <td>0.05</td>
      <td>78</td>
    </tr>
    <tr>
      <th>19534</th>
      <td>USC00516128</td>
      <td>2017-08-08</td>
      <td>0.34</td>
      <td>74</td>
    </tr>
    <tr>
      <th>19535</th>
      <td>USC00516128</td>
      <td>2017-08-09</td>
      <td>0.15</td>
      <td>71</td>
    </tr>
    <tr>
      <th>19536</th>
      <td>USC00516128</td>
      <td>2017-08-10</td>
      <td>0.07</td>
      <td>75</td>
    </tr>
    <tr>
      <th>19538</th>
      <td>USC00516128</td>
      <td>2017-08-12</td>
      <td>0.14</td>
      <td>74</td>
    </tr>
    <tr>
      <th>19540</th>
      <td>USC00516128</td>
      <td>2017-08-14</td>
      <td>0.22</td>
      <td>79</td>
    </tr>
    <tr>
      <th>19541</th>
      <td>USC00516128</td>
      <td>2017-08-15</td>
      <td>0.42</td>
      <td>70</td>
    </tr>
    <tr>
      <th>19542</th>
      <td>USC00516128</td>
      <td>2017-08-16</td>
      <td>0.42</td>
      <td>71</td>
    </tr>
    <tr>
      <th>19543</th>
      <td>USC00516128</td>
      <td>2017-08-17</td>
      <td>0.13</td>
      <td>72</td>
    </tr>
    <tr>
      <th>19545</th>
      <td>USC00516128</td>
      <td>2017-08-19</td>
      <td>0.09</td>
      <td>71</td>
    </tr>
    <tr>
      <th>19547</th>
      <td>USC00516128</td>
      <td>2017-08-21</td>
      <td>0.56</td>
      <td>76</td>
    </tr>
    <tr>
      <th>19548</th>
      <td>USC00516128</td>
      <td>2017-08-22</td>
      <td>0.50</td>
      <td>76</td>
    </tr>
    <tr>
      <th>19549</th>
      <td>USC00516128</td>
      <td>2017-08-23</td>
      <td>0.45</td>
      <td>76</td>
    </tr>
  </tbody>
</table>
<p>18103 rows × 4 columns</p>
</div>




```python
clean_raw_measurements_df.shape
```




    (18103, 4)




```python
#Find duplicate rows
search = pd.DataFrame.duplicated(raw_stations_df)
search[search == True]
```




    Series([], dtype: bool)



No duplicate rows found
Look for missing data


```python
raw_stations_df.shape
```




    (9, 5)




```python
raw_stations_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 9 entries, 0 to 8
    Data columns (total 5 columns):
    station      9 non-null object
    name         9 non-null object
    latitude     9 non-null float64
    longitude    9 non-null float64
    elevation    9 non-null float64
    dtypes: float64(3), object(2)
    memory usage: 440.0+ bytes
    


```python
raw_stations_df.describe()
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
      <th>latitude</th>
      <th>longitude</th>
      <th>elevation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>9.000000</td>
      <td>9.000000</td>
      <td>9.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>21.393826</td>
      <td>-157.867098</td>
      <td>60.977778</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.086442</td>
      <td>0.103873</td>
      <td>103.465547</td>
    </tr>
    <tr>
      <th>min</th>
      <td>21.271600</td>
      <td>-158.011100</td>
      <td>0.900000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>21.333100</td>
      <td>-157.975100</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>21.393400</td>
      <td>-157.837400</td>
      <td>14.600000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>21.451670</td>
      <td>-157.802500</td>
      <td>32.900000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>21.521300</td>
      <td>-157.711390</td>
      <td>306.600000</td>
    </tr>
  </tbody>
</table>
</div>



Stations Data looks clean


```python
#Write clean data to new CSV files
clean_raw_measurements_df.to_csv('clean_hawaii_measurements.csv')
raw_stations_df.to_csv('clean_hawaii_stations.csv')
```
