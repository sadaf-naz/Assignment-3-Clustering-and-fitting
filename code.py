# import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def read_worldbank(filename: str):
  """
    Reads a file containing world bank data and returns the original dataframe, the dataframe with countries as columns, and the dataframe with year as columns.

    Parameters:
    - filename (str): The name of the file to be read, including the file path.

    Returns:
    - dataframe (pandas dataframe): The original dataframe containing the data from the file.
    - df_transposed_country (pandas dataframe): Dataframe with countries as columns.
    - df_transposed_year (pandas dataframe): Dataframe with year as columns.
  """
  # Read the file into a pandas dataframe
  dataframe = pd.read_csv(filename)

  # Transpose the dataframe
  df_transposed = dataframe.transpose()

  # Populate the header of the transposed dataframe with the header information

  # silice the dataframe to get the year as columns
  df_transposed.columns = df_transposed.iloc[1]

  # As year is now columns so we don't need it as rows
  df_transposed_year = df_transposed[0:].drop('year')

  # silice the dataframe to get the country as columns
  df_transposed.columns = df_transposed.iloc[0]

  # As country is now columns so we don't need it as rows
  df_transposed_country = df_transposed[0:].drop('country')

  return dataframe, df_transposed_country, df_transposed_year

# load data from World Bank website or a similar source
df, df_country, df_year = read_worldbank('worldbank_dataset.csv')

def remove_null_values(feature):
  """
  This function removes null values from a given feature.


  Parameters:
    feature (pandas series): The feature to remove null values from.

  Returns:
    numpy array: The feature with null values removed.
  """
  # drop null values from the feature
  return np.array(feature.dropna())

df.columns[2:]

def balance_data(df):
  """
  This function takes a dataframe as input and removes missing values from each column individually.
  It then returns a balanced dataset with the same number of rows for each column.

  Input:

  df (pandas dataframe): a dataframe containing the data to be balanced
  Output:

  balanced_df (pandas dataframe): a dataframe with the same number of rows for each column,
   after removing missing values from each column individually
  """
  # Making dataframe of all the feature in the avaiable in
  # dataframe passing it to remove null values function
  # for dropping the null values

  forest_area = remove_null_values(df[['forest_area']])

  agricultural_land = remove_null_values(df[['agricultural_land']])

  nitrous_oxide = remove_null_values(df[['nitrous_oxide']])

  min_length = min(len(forest_area), len(agricultural_land), len(nitrous_oxide))

   # after removing the null values we will create datafram

  clean_data = pd.DataFrame({
                                'country': [df['country'].iloc[x] for x in range(min_length)],
                                'year': [df['year'].iloc[x] for x in range(min_length)],
                                'forest_area': [forest_area[x][0] for x in range(min_length)],
                                'agricultural_land': [agricultural_land[x][0] for x in range(min_length)],
                                 'nitrous_oxide': [nitrous_oxide[x][0] for x in range(min_length)]
                                 })
  return clean_data

# Clean and preprocess the data
df = balance_data(df)

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['forest_area', 'agricultural_land',
       'nitrous_oxide']])

# Cluster analysis
c = 5
kmeans = KMeans(n_clusters=c)
kmeans.fit(scaled_data)
df['cluster'] = kmeans.labels_

# Function for scatter plot visualization
def plot_clusters(feature1, feature2):
    plt.figure(figsize=(8, 6))
    for i in range(c):
        cluster_data = df[df['cluster'] == i]
        plt.scatter(cluster_data[feature1], cluster_data[feature2], label=f'Cluster {i}')

    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', c='black', label='Cluster Centers')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title(f'Cluster Membership based on {feature1} vs {feature2}')
    plt.legend()
    plt.show()

# Visualize clusters for different attribute combinations
plot_clusters('forest_area', 'nitrous_oxide')

plot_clusters('forest_area', 'agricultural_land')

# Function for country-specific cluster plot
def plot_country_cluster(country_name, feature1, feature2):
    country_data = df[df['country'] == country_name]
    plt.figure(figsize=(8, 6))
    for i in range(c):
        cluster_data = country_data[country_data['cluster'] == i]
        plt.scatter(cluster_data[feature1], cluster_data[feature2], label=f'Cluster {i}')

    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', c='black', label='Cluster Centers')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title(f'Cluster Membership for {country_name}')
    plt.legend()
    plt.show()

df.columns

df.country.unique()

# Visualize clusters for specific countries
plot_country_cluster('Mongolia', 'forest_area', 'nitrous_oxide')

plot_country_cluster('Central African Republic', 'forest_area', 'agricultural_land')

def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.

    This routine can be used in assignment programs.
    """

    import itertools as iter

    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower

    uplow = []   # list to hold upper and lower limits for parameters
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))

    pmix = list(iter.product(*uplow))

    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)

    return lower, upper

# Define the exponential function
def exp_func(x, a, b):
    return a * np.exp(b * x)

df.columns

# Data of cluster 4
c1 = df[(df['cluster'] == 2)]

# x values and y values
x = c1['forest_area']
y = c1['nitrous_oxide']

popt, pcov = curve_fit(exp_func, x, y)

# Use err_ranges function to estimate lower and upper limits of the confidence range
sigma = np.sqrt(np.diag(pcov))
lower, upper = err_ranges(x, exp_func, popt,sigma)

# Use pyplot to create a plot showing the best fitting function and the confidence range
plt.plot(x, y, 'o', label='data')
plt.plot(x, exp_func(x, *popt), '-', label='fit')
plt.fill_between(x, lower, upper, color='pink', label='confidence interval')
plt.legend()
plt.xlabel('forest_area')
plt.ylabel('nitrous_oxide')
plt.show()

# Define the range of future x-values for which you want to make predictions
future_x = np.arange(90,95)

# Use the fitted function and the estimated parameter values to predict the future y-values
future_y = exp_func(future_x, *popt)

# Plot the predictions along with the original data
plt.plot(x, y, 'o', label='data')
plt.plot(x, exp_func(x, *popt), '-', label='fit')
plt.plot(future_x, future_y, 'o', label='future predictions')
plt.xlabel('forest_area')
plt.ylabel('nitrous_oxide')
plt.legend()
plt.show()