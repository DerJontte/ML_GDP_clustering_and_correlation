# Script to calculate and visualize correlations between GDP or cluster membership with other indicators for a given group of countries.
#
# (c) 2019 John Nordqvist
# Licence: MIT
#

import pandas as pd
import dataset_handling as dataset
import kmeans_by_gdp as clustering


def main():
    # Load the dataset
    print("Loading data... ", end='')
    data = pd.read_csv('API_19_DS2_en_csv_v2_10400593.csv', skiprows=4)
    print("done.")

    correlate_to = 'GDP'  # Choose which variable to calculate correlations for, 'GDP' or 'cluster'

    data = dataset.prune_data(data)  # Remove rows and columns that will not be used

    indicator_name = data.pop('Indicator Name')  # Store the indicator names for later use and remove them from the data (for the clustering to work)
    changed_countries, gdp_per_capita_all_countries = clustering.main(data=data)  # Get a list of countries that have changed from one group to another regarding GDP over the years 1960-2014
    data['Indicator Name'] = indicator_name  # Restore the indicator names to the data

    # These three countries were chosen maually after inspection of their change in clusters and considering some
    # traits in their economy. Change the array to the part that in the comment at the end of the line to get all
    # countries that have changed clusters over the years(and that have complete GDP data 1960-2014).
    countries_of_interest = ['Singapore', 'Venezuela, RB', 'Bahamas, The'] # changed_countries.index.to_list()
    output_string = ''

    # Iterate over the countries defined above and create an output with select data
    for country in countries_of_interest:
        gdp_per_capita = dataset.get_country_by_index(gdp_per_capita_all_countries, country)
        # Country name and cluster belonging over the years
        output_string += "\n{}\n". format(country)
        output_string += "Clusters 1960-2014: {}\n".format(str(changed_countries[changed_countries.index == country].values[0]).replace('\n',''))
        output_string += "GDP 1960: {:.2f}\tGDP 2014: {:.2f}\tCorrelating to: {}\n".format(gdp_per_capita['1960'][0], gdp_per_capita['2014'][0], correlate_to)

        # Iterate over all variables in the dataset for the current country and process those with complete timeseries.
        # The correlation can be calculated for either GDP per capita or cluster belonging. The GDP is a continous variable,
        # and correlation is thus calculated with Pearson's r. Cluster belonging is a non-normally distributed ordinal scale,
        # for which correlation is calculated with Spearman's rank.
        for code in data['Indicator Code'].unique():
            if correlate_to == 'GDP':
                from scipy.stats.stats import pearsonr as get_correlation
                correlatable = dataset.get_country_by_index(gdp_per_capita_all_countries, country)
            elif correlate_to == 'cluster':
                correlatable = changed_countries[changed_countries.index == country]
                correlatable.rename(str, axis='columns', inplace=True)
                from scipy.stats.stats import spearmanr as get_correlation

            correlatable_reshaped = correlatable.values.reshape(-1, )

            current = dataset.get_indicator_code(data, code)  # Filter the data for the indicator
            indicator_name = current['Indicator Name'][current.index[0]]  # Store the indicator name
            current = dataset.clean_matrix_incomplete(current)
            current = dataset.get_country_by_index(current, country)  # Keep data only for the current country
            current = dataset.clean_matrix(current)
            current_reshaped = current.values.reshape(-1, )

            if len(current_reshaped) != len(correlatable_reshaped):  # Remove years that are missing from the indicator from GDP as well
                for year_int in range(1960,2015):
                    year = str(year_int)
                    if year not in current.keys():
                        correlatable.pop(year) if year in correlatable.keys() else None
                correlatable_reshaped = correlatable.values.reshape(-1, )
                current_reshaped = current.values.reshape(-1, )

            if len(current_reshaped) == len(correlatable_reshaped):
                correlation, p_value = get_correlation(current_reshaped, correlatable_reshaped)  # Calcultae correlation between variable and GDP per capita or cluster belonging
                if not pd.isna(correlation):  # If the correlation is a number (i.e. no errors took place), add it to the output
                    sig = '*' if p_value < 0.05 else ''
                    p_value = "{:7.3f}".format(p_value) if p_value >= 0.001 else '< 0.001'
                    output_string += "\t{:70.67}Correlation: {:5.2f}\t\t p-value: {:11}{}\t\tn={}\n".format(indicator_name, correlation, p_value, sig, len(current_reshaped))

    print(output_string)  # Print the resulting list to the console (TODO: create a heatmap with pyplot)


if __name__ == "__main__":
    main()
