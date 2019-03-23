from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dataset_handling as dataset
import kmeans_by_gdp as clustering
from scipy.stats.stats import pearsonr


def main():
    print("Loading data... ", end='')
    data = pd.read_csv('API_19_DS2_en_csv_v2_10400593.csv', skiprows=4)
    print("done.")

    data = dataset.prune_data(data)

    indicator_name = data.pop('Indicator Name')
    changed_countries, gdp_per_capita_all_countries = clustering.main(data=data)
    data['Indicator Name'] = indicator_name

    countries_of_interest = ['Singapore', 'Venezuela, RB', 'Bahamas, The'] # changed_countries.index.to_list()
    correlation_list = ''

    for country in countries_of_interest:
        correlation_list += "\n{}\nClusters 1960-2014: {}\n".format(country, str(changed_countries[changed_countries.index == country].values[0]).replace('\n',''))
        gdp_per_capita = dataset.get_country_by_index(gdp_per_capita_all_countries, country)
        gdp_per_capita = gdp_per_capita.values.reshape(-1,)

        for code in data['Indicator Code'].unique():
            current = dataset.get_indicator_code(data, code)
            indicator_name = current['Indicator Name'][current.index[0]]
            current = dataset.clean_matrix_incomplete(current)
            current = dataset.get_country_by_index(current, country)
            current = current.values.reshape(-1, )
            if len(current) == len(gdp_per_capita):
                correlation, p_value = pearsonr(current, gdp_per_capita)
                if not pd.isna(correlation):
                    sig = '*' if p_value < 0.05 else ''
                    p_value = "{:8.2f}".format(p_value) if p_value >= 0.0001 else '< 0.0001'
                    correlation_list += "\t{:70.67}Correlation: {:5.2f}\t\t p-value: {:11} {}\n".format(indicator_name, correlation, p_value, sig)

    print(correlation_list)


if __name__ == "__main__":
    main()
