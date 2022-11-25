import pandas as pd
from numpy import dtype


def to_customers(transactions: pd.DataFrame, key: list[str, ...]) -> pd.DataFrame:
    """
    Function that gets the DataFrame with the transactions and returns the DataFrame
    holding the information about the preprocessed set of customers.
    The input must respect the schema hereby specified:

    - CustomerID ->                 object
    - CustomerDOB ->         datetime64[ns]
    - CustGender   ->               object
    - CustLocation   ->             object
    - CustAccountBalance ->        float64
    - TransactionDate ->     datetime64[ns]
    - TransactionTime  ->            int64
    - TransactionAmount (INR) ->   float64

    :param transactions: The DataFrame holding the transaction data.
    :param key: The key attributes used in order to group transactions.
    :return: The DataFrame holding the attributes per single customer.
    """

    # Check the input schema, if not the correct one -> AssertionError
    assert (transactions.columns.tolist() == ['CustomerID', 'CustomerDOB', 'CustGender',
                                              'CustLocation', 'CustAccountBalance', 'TransactionDate',
                                              'TransactionTime', 'TransactionAmount (INR)']
            and ['datetime' if pd.api.types.is_datetime64_any_dtype(x) else x for x in transactions.dtypes.tolist()]
            == [dtype('O'), 'datetime', dtype('O'), dtype('O'), dtype('float64'),
                'datetime', dtype('int64'), dtype('float64')]), \
        'This function expects a certain input schema, as written in its documentation/description.'

    transactions = transactions.drop('TransactionTime', axis=1)  # Remove Linux timestamp

    # Aggregate in order to set the resolution of the dataframe at customer level
    grouped = transactions.groupby(key)

    # Apply different aggregation functions to different columns
    # Notice that to build the customer dataset we are considering the last recorded account balance
    customers = pd.concat([grouped.agg(dict([
        ('CustLocation', pd.Series.mode), ('TransactionDate', pd.Series.max),
        ('TransactionAmount (INR)', pd.Series.mean)])),
        transactions.loc[grouped.TransactionDate.idxmax()].set_index(key)['CustAccountBalance']], axis=1, copy=False)

    # Reset the index (set a new customer identifier) and extract the MultiIndex
    customers.reset_index(inplace=True, names=key)
    # Remove customer ID since we do not use that for the clustering
    customers.drop("CustomerID", inplace=True, axis=1)

    return customers


if __name__ == '__main__':
    import dotenv
    import os

    dotenv.load_dotenv('../../ext_variables.env')
    file_path = os.path.join(os.getenv("PATH_FILES_ADM"), 'bank_transactions.csv')
    transaction_table = pd.read_csv(file_path, parse_dates=['CustomerDOB', 'TransactionDate'],
                                    infer_datetime_format=True, index_col='TransactionID', nrows=10000)
    test_to_custom = to_customers(transaction_table, key=['CustomerID', 'CustomerDOB', 'CustGender'])
