import pandas as pd
from numpy import dtype
import numpy as np
import scipy.sparse as sparse
import sklearn.preprocessing as sk_pre


def clean_time(time_data: pd.Series, timestamp: bool, old_dates: bool) -> pd.Series:
    """
    Function to clean time data, repairing the string format of dates and day timestamps.

    :param time_data: Series with the datetime values to clean.
    :param timestamp: Whether we are dealing with the day timestamp from TransactionTime or with dates.
    :param old_dates: Whether we are dealing with the date of birth or with the date of the transaction.
    :return: Cleaned series
    """
    time_data = time_data.copy()
    if not timestamp:
        if old_dates:
            # DOB from 1800 considered as nan
            time_data[time_data == '1/1/1800'] = np.nan
        # add 19 or 20 to dates in order to avoid problems in parsing the year
        time_data[~time_data.isna()] = \
            time_data[~time_data.isna()].str.replace(r'(?=\d{2}$)', '19' if old_dates else '20', regex=True)
        time_data = pd.to_datetime(time_data, errors='coerce', dayfirst=True)
    else:
        # add 0 in order to avoid problems in parsing the hour
        time_data = pd.to_datetime(time_data.astype(str).str.zfill(6), format='%H%M%S')
    return time_data


def get_age(dob: pd.Series) -> pd.Series:
    """
    Function to get the age of a customer starting from its date of birth.

    :param dob: Series with the date of birth for each customer
    :return: Series with the age (int) for each customer
    """
    # Get age through floor division
    age = (pd.Timestamp.now() - dob.fillna(dob.mean())) // np.timedelta64(1, 'Y')
    # age greater than 100 comes from having wrongly added 19 in place of 20, reverse it
    age[age >= 100] = age[age >= 100] - 100
    return age


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
    - TransactionTime  ->    datetime64[ns]
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
                'datetime', 'datetime', dtype('float64')]), \
        'This function expects a certain input schema, as written in its documentation/description.'

    # Aggregate in order to set the resolution of the dataframe at customer level
    grouped = transactions.groupby(key)

    # Apply different aggregation functions to different columns
    # Notice that to build the customer dataset we are considering the most recently recoded account balance
    customers = pd.concat([grouped.agg(dict([
        ('CustLocation', lambda x: x.mode().iat[0] if np.sum(~x.isna()) else np.nan), ('TransactionTime', 'mean'),
        ('TransactionAmount (INR)', 'mean')])),
        transactions.loc[grouped.TransactionDate.idxmax()].set_index(key)['CustAccountBalance']], axis=1, copy=False)

    # Impute NAs for Customer Location
    customers.CustLocation.fillna(customers.CustLocation.mode().iat[0], inplace=True)

    # Get only the hour from Transaction Time after the aggregation mean
    customers.TransactionTime = customers.TransactionTime.dt.hour
    # Reset the index (set a new customer identifier) and extract the MultiIndex
    customers.reset_index(inplace=True, names=key)
    # Remove customer ID since we do not use that for the clustering
    customers.drop("CustomerID", inplace=True, axis=1)

    return customers


class Shingling:
    """
    Class keeping the state of the shingling transformation. It holds the scikit object used to create the shingle
    matrix and needed to reproduce the transformation for new query observations. Initialization builds the
    (SciPy) sparse shingle matrix from the customer data, while new transformations can be performed with
    the 'transform' method.

    Constructor factory method works directly on the transaction data in order to build the object.
    """

    # The init method is just the initialization of a sequence of KBinsDiscretizer and OneHotEncoder scikit objects
    def __init__(self, customers: pd.DataFrame) -> None:
        self.costumer_table = customers  # Save customer table to return results for the query

        # Age
        self.age_encoder = \
            sk_pre.KBinsDiscretizer(15, encode='onehot-dense', strategy='quantile',
                                    subsample=200000).fit(np.expand_dims(customers.age, 1))
        one_hot_age = self.age_encoder.transform(np.expand_dims(customers.age, 1))

        # Gender
        self.gender_encoder = sk_pre.OneHotEncoder(sparse=False).fit(np.expand_dims(customers.CustGender, 1))
        one_hot_gender = self.gender_encoder.transform(np.expand_dims(customers.CustGender, 1))

        # Transaction Time
        self.time_trans_encoder = \
            sk_pre.KBinsDiscretizer(24, encode='onehot-dense', strategy='uniform',
                                    ).fit(np.expand_dims(customers.TransactionTime, 1))
        one_hot_time_trans = self.time_trans_encoder.transform(np.expand_dims(customers.TransactionTime, 1))

        # Transaction Amount
        self.trans_amount_encoder = \
            sk_pre.KBinsDiscretizer(20, encode='onehot-dense', strategy='quantile', subsample=200000).fit(
                np.expand_dims(customers['TransactionAmount (INR)'], 1))
        one_hot_amount = self.trans_amount_encoder.transform(np.expand_dims(customers['TransactionAmount (INR)'], 1))

        # Account balance
        self.account_balance_encoder = sk_pre.KBinsDiscretizer(20, encode='onehot-dense', strategy='quantile',
                                                               subsample=200000).fit(
            np.expand_dims(customers.CustAccountBalance, 1))
        one_hot_balance = self.account_balance_encoder.transform(np.expand_dims(customers.CustAccountBalance, 1))

        # Common Customer Location
        self.location_encoder = sk_pre.OneHotEncoder(sparse=False, max_categories=10,
                                                     handle_unknown='infrequent_if_exist')\
            .fit(np.expand_dims(customers.CustLocation, 1))
        one_hot_location = self.location_encoder.transform(np.expand_dims(customers.CustLocation, 1))

        self.shingle_matrix = sparse.csr_matrix(np.concatenate([one_hot_age, one_hot_gender,
                                                                one_hot_time_trans, one_hot_amount, one_hot_balance,
                                                                one_hot_location], axis=1).T)

    def transform(self, customer_query: pd.DataFrame) -> sparse.csr_matrix:
        """
        Method to do the shingling transformation on query observations.
        :param customer_query: A Pandas DataFrame with the query observations
        :return: A shingle matrix representing the query observations.
        """
        customer_query = customer_query.copy()
        # Clean DOBs
        customer_query.CustomerDOB = clean_time(customer_query.CustomerDOB, timestamp=False, old_dates=True)
        # Clean transaction times
        customer_query.TransactionTime = clean_time(customer_query.TransactionTime, timestamp=True,
                                                    old_dates=False)
        # Clean transaction dates
        customer_query.TransactionDate = clean_time(customer_query.TransactionDate, timestamp=False,
                                                    old_dates=False)
        # Get age from DoB
        customer_query['age'] = get_age(customer_query.CustomerDOB)

        shingle_matrix_query = sparse.csr_matrix(np.concatenate([
            self.age_encoder.transform(np.expand_dims(customer_query.age, 1)),
            self.gender_encoder.transform(np.expand_dims(customer_query.CustGender, 1)),
            self.time_trans_encoder.transform(np.expand_dims(customer_query.TransactionTime, 1)),
            self.trans_amount_encoder.transform(np.expand_dims(customer_query['TransactionAmount (INR)'], 1)),
            self.account_balance_encoder.transform(np.expand_dims(customer_query.CustAccountBalance, 1)),
            self.location_encoder.transform(np.expand_dims(customer_query.CustLocation, 1))], axis=1).T)
        return shingle_matrix_query

    @classmethod
    def constructor(cls, transaction_table: pd.DataFrame) -> "Shingling":
        """
        Factory method to build an instance of Shingling from dirty transaction data.

        :param transaction_table: Table with the information for each transaction.
        :return: instance of the Shingling class.
        """
        # With deep copy the underlying Numpy array IS NOT copied
        # Copying the object prevents side effects though
        transaction_table = transaction_table.copy()
        # Clean DOBs
        transaction_table.CustomerDOB = clean_time(transaction_table.CustomerDOB, timestamp=False, old_dates=True)
        # Clean Transaction Times
        transaction_table.TransactionTime = clean_time(transaction_table.TransactionTime, timestamp=True,
                                                       old_dates=False)
        # Clean Transaction Dates
        transaction_table.TransactionDate = clean_time(transaction_table.TransactionDate, timestamp=False,
                                                       old_dates=False)
        # Impute NAs in Balance
        transaction_table.loc[transaction_table.CustAccountBalance.isna(), 'CustAccountBalance'] = \
            transaction_table.CustAccountBalance.mean()
        # Get customer data
        customers = to_customers(transaction_table, key=['CustomerID', 'CustomerDOB', 'CustGender'])
        # Get age from DOBs
        customers['age'] = get_age(customers.CustomerDOB)
        # Drop IDs from the original df
        customers.drop('CustomerDOB', axis=1, inplace=True)

        return cls(customers)
