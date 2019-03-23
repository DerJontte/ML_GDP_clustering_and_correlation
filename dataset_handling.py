# Function to pick out one single indicator code into an own matrix
def get_indicator_code(data, indicator_code):
    print('Filtering for indicator code {}... '.format(indicator_code), end='')
    to_return = data[data['Indicator Code'] == indicator_code]
    print('done.')
    return to_return


# Function get the data for a given country when the country names are in a column in the data
def get_country(data, country):
    return data.loc[data['Country Name'] == country]


# Function to get the data for a given country when country names are indices
def get_country_by_index(data, country):
    return data[data.index == country]


# Function to format a matrix to contain only usable data
def clean_matrix(data):
    data = clean_matrix_incomplete(data)
    data.dropna(axis='index', how='any', inplace=True)  # Remove incomplete rows
    return data


# Function to format a matrix to contain only usable data but allowing incomplete rows
def clean_matrix_incomplete(data):
    if 'Country Name' in data:
        data.rename(index=data['Country Name'], inplace=True)  # Index the data by country name
        data.pop('Country Name')  # Remove column with country name
    if 'Indicator Code' in data:
        data.pop('Indicator Code')  # Remove column with indicator code
    if 'Indicator Name' in data:
        data.pop('Indicator Name')
    data.dropna(axis='columns', how='all', inplace=True)  # Remove columns that contain no data at all
    return data


# Function to remove unneeded data from the dataset
def prune_data(data):
    print("Removing unneded and incomplete data... ", end='')
    data = data[~data['Country Name'].str.contains("World")]
    data = data[~data['Country Name'].str.contains("IDA")]
    data = data[~data['Country Name'].str.contains("HIPC")]
    data = data[~data['Country Name'].str.contains("IBRD")]
    data = data[~data['Country Name'].str.contains("Sub-Saharan")]
    data = data[~data['Country Name'].str.contains("Central Europe and the Baltics")]
    data = data[~data['Country Name'].str.contains("Central Asia")]
    data = data[~data['Country Name'].str.contains("Latin America & Caribbean")]
    data = data[~data['Country Name'].str.contains("Middle East")]
    data = data[~data['Country Name'].str.contains("East Asia")]
    data = data[~data['Country Name'].str.contains("North America")]
    data = data[~data['Country Name'].str.contains("European Union")]
    data = data[~data['Country Name'].str.contains("mall states")]
    data = data[~data['Country Name'].str.contains("members")]
    data = data[~data['Country Name'].str.contains("classification")]
    data = data[~data['Country Name'].str.contains("situations")]
    data = data[~data['Country Name'].str.contains("income")]
    data = data[~data['Country Name'].str.contains("dividend")]
    data = data[~data['Country Name'].str.contains("classified")]
    data = data[~data['Country Name'].str.contains("Euro area")]
    data.pop('Country Code')
    data.pop('2015')
    data.pop('2016')
    data.pop('2017')
    data.pop('2018')
    print("done.")
    return data
