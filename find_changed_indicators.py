import pandas as pd
import dataset_handling as dataset
import kmeans_by_gdp as clustering
from scipy.stats.stats import pearsonr


def main():
    # Load the dataset
    print("Loading data... ", end='')
    data = pd.read_csv('API_19_DS2_en_csv_v2_10400593.csv', skiprows=4)
    print("done.")

    data = dataset.prune_data(data)  # Remove rows and columns that will not be used

    indicator_name = data.pop('Indicator Name')  # Store the indicator names for later use and remove them from the data (for the clustering to work)
    changed_countries, gdp_per_capita_all_countries = clustering.main(data=data)  # Get a list of countries that have changed from one group to another regarding GDP over the years 1960-2014
    data['Indicator Name'] = indicator_name  # Restore the indicator names to the data

    # These three countries were chosen maually after inspection of their change in clusters and considering some
    # traits in their economy. Change the array to the part that in the comment at the end of the line to get all
    # countries that have changed clusters over the years(and that have complete GDP data 1960-2014).
    countries_of_interest = ['Singapore', 'Venezuela, RB', 'Bahamas, The'] # changed_countries.index.to_list()
    correlation_list = ''

    # Iterate over the countries defined above and create an output with select data
    for country in countries_of_interest:
        # Country name and cluster belonging over the years
        correlation_list += "\n{}\nClusters 1960-2014: {}\n".format(country, str(changed_countries[changed_countries.index == country].values[0]).replace('\n',''))

        # Iterate over all variables in the dataset for the current country and process those with complete timeseries
        for code in data['Indicator Code'].unique():
            gdp_per_capita = dataset.get_country_by_index(gdp_per_capita_all_countries, country)
            gdp_per_capita_reshaped = gdp_per_capita.values.reshape(-1, )

            current = dataset.get_indicator_code(data, code)  # Filter the data for the indicator
            indicator_name = current['Indicator Name'][current.index[0]]  # Store the indicator name
            current = dataset.clean_matrix_incomplete(current)
            current = dataset.get_country_by_index(current, country)  # Keep data only for the current country
            current_reshaped = current.values.reshape(-1, )

            if len(current_reshaped) != len(gdp_per_capita_reshaped):  # Remove years that are missing from the indicator from GDP as well
                for year_int in range(1960,2015):
                    year = str(year_int)
                    if year not in current.keys():
                        gdp_per_capita.pop(year) if year in gdp_per_capita.keys() else None
                gdp_per_capita_reshaped = gdp_per_capita.values.reshape(-1, )
                current_reshaped = current.values.reshape(-1, )

            if len(current_reshaped) == len(gdp_per_capita_reshaped):
                correlation, p_value = pearsonr(current_reshaped, gdp_per_capita_reshaped)  # Calcultae Pearson's correlation between variable and GDP per capita
                if not pd.isna(correlation):  # If the correlation is a number (i.e. no errors took place), add it to the output
                    sig = '*' if p_value < 0.05 else ''
                    p_value = "{:8.2f}".format(p_value) if p_value >= 0.0001 else '< 0.0001'
                    correlation_list += "\t{:70.67}Correlation: {:5.2f}\t\t p-value: {:11}{}\t\tn={}\n".format(indicator_name, correlation, p_value, sig, len(current_reshaped))

    print(correlation_list)  # Print the resulting list to the console (TODO: create a heatmap with pyplot)


if __name__ == "__main__":
    main()
