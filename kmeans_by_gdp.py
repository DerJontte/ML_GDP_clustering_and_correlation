# The sorting of clusters from lowest to highest income is taken from the stackoverflow-topic at the address
# https://stackoverflow.com/questions/44888415/how-to-set-k-means-clustering-labels-from-highest-to-lowest-with-python

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import dataset_handling as dataset

pd.options.mode.chained_assignment = None
k = 6  # Number of clusters
groups = 3  # The final number of categories to group the clusters into


def main(data=None):
    if data is None:
        print("Loading data... ", end='')
        data = pd.read_csv('API_19_DS2_en_csv_v2_10400593.csv', skiprows=4)
        print("done.")

        data = dataset.prune_data(data)  # Remove country groups, years without data and unnecessary columns from the data
        indicator_name = data.pop('Indicator Name')

    # ## Create some matrices that will be used for calculating the GDP per capita for each year ## #

    co2_per_usd_gdp = dataset.get_indicator_code(data, 'EN.ATM.CO2E.KD.GD')  # CO2 emissions in kg per 2010 US$ of GDP
    co2_per_usd_gdp = dataset.clean_matrix(co2_per_usd_gdp)

    co2_pc = dataset.get_indicator_code(data, 'EN.ATM.CO2E.PC')  #CO2 emissions in metric tons per capita
    co2_pc = dataset.clean_matrix(co2_pc)

    co2_kg = co2_pc * 1000  # CO2 emissions in kilograms

    # ## Matrices done ## #

    gdp_per_capita = pd.DataFrame()  # Initialize a dataframe to store the GDP per capita series in

    # Loop through the counrty names in co2_kg, and calculate the GDP per capita for those countries that have
    # a row in the co2_per_usd_gdp as well.
    print("Calculating GDP per capita... ", end='')
    for index in co2_kg.index:
        if index in co2_per_usd_gdp.index:
            gdp_per_capita = gdp_per_capita.append(co2_kg[co2_kg.index == index] / co2_per_usd_gdp[co2_per_usd_gdp.index == index])
    print("done.")

    centers_ordered_timeseries = []  # Create arrays for the timeseries produced by the kmeans-clusterings
    labels_ordered_timeseries = []

    # ## Perform k-means clustering for each year there is data for and save the results in th arrays create above ## #
    for year in gdp_per_capita.keys():
        kmeans = KMeans(n_clusters=k, random_state=42).fit(gdp_per_capita[year].__array__().reshape(-1, 1))

        # Recode the clusters so that the cluster numbers correspond to the relative wealth of the countries,
        # with 0 being those with the smallest GDP and k those with the highest GDP.
        idx = np.argsort(kmeans.cluster_centers_.sum(axis=1), axis=0)
        lut = np.zeros_like(idx)
        lut[idx] = np.arange(k)

        # Flatten the array with cluster centers to one dimension and store it
        centers_ordered = kmeans.cluster_centers_[idx].sum(axis=1)
        centers_ordered_timeseries.append(centers_ordered)

        # Downsample the clustering to three clusters with low, middle and high income countries respectively
        labels_ordered = (lut[kmeans.labels_] / (k / groups)).astype(int)
        labels_ordered_timeseries.append(labels_ordered)

    # Make a numpy.array with the transposed clustering data to get per-country timeseries, print out the countries
    # that have changed clusters over the years (or return an array with said countries to the caller).
    country_timeseries = np.array(labels_ordered_timeseries).T
    changed_countries = pd.DataFrame()
    for i in range(0, len(country_timeseries)):
        if(country_timeseries[i] != country_timeseries[i][0]).any():
            country = gdp_per_capita.index[i]
            timeseries = pd.DataFrame(country_timeseries[i]).T
            timeseries.rename(index={0: country}, inplace=True)  # Index the data by country name
            changed_countries = changed_countries.append(timeseries)

    # Rename the columns of changed_countries to the years they represent
    years_dict = {}
    for key in changed_countries.keys():
        years_dict.update({key: key + 1960})
    changed_countries.rename(columns=years_dict, inplace=True)

    return changed_countries, gdp_per_capita


# Entry point when the this script is executed independently
if __name__ == "__main__":
    changed, gdp_pc = main()
    print(changed.to_string())