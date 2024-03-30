import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import timeit
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore")
regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'


def dcshape(X_train, X_test, y_train, y_test):

    """
    Display the shape (number of rows and columns) of input arrays.

    Parameters:
    - X_train (array-like): Training input features.
    - X_test (array-like): Testing input features.
    - y_train (array-like): Training target labels.
    - y_test (array-like): Testing target labels.

    Returns:
    None

    Example:
    >>> X_train_data = [[1, 2], [3, 4]]
    >>> X_test_data = [[5, 6], [7, 8]]
    >>> y_train_data = [0, 1]
    >>> y_test_data = [1, 0]
    >>> dcshape(X_train_data, X_test_data, y_train_data, y_test_data)
    The rows and columns in X_train are : (2, 2)
    The rows and columns of X_test are: (2, 2)
    The rows and columns of y_train are: (2,)
    The rows and columns of y_test are: (2,)
    """
    print("The rows and columns in X_train are :\t", X_train.shape)
    print("The rows and columns of X_test are:\t", X_test.shape)
    print("The rows and columns of y_train are:\t", y_train.shape)
    print("The rows and columns of y_test are:\t", y_test.shape)

    
def dcsample(df, rw=10):
    """
    Randomly samples rows from a DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame from which rows will be sampled.
    - rw (int, optional): The number of rows to sample. Default is 10. If None is provided, it defaults to 10.

    Returns:
    pd.DataFrame: A DataFrame containing randomly sampled rows from the input DataFrame.

    Example:
    >>> import pandas as pd
    >>> data = {'feature1': [1, 2, 3, 4, 5], 'feature2': [6, 7, 8, 9, 10]}
    >>> df = pd.DataFrame(data)
    >>> sampled_df = dcsample(df, rw=3)
    >>> print(sampled_df)
       feature1  feature2
    1         2         7
    3         4         9
    0         1         6
    """
    import pandas as pd
    
    if rw is None:
        rw = 10
    
    return df.sample(rw)

def dc_describe_categorical(X):
    """
    Just like .decribe(), but returns the results for categorical variables only.
    """
    from IPython.display import display, HTML
    display(HTML(X[X.columns[X.dtypes == "object"]].describe().to_html()))


def dc_numericalCategoricalSplit(df):
    """
    Split a DataFrame into numerical and categorical data.

    Parameters:
    df : pandas.DataFrame
        The input DataFrame to be split.

    Returns:
    numerical_data : pandas.DataFrame
        DataFrame containing only the numerical features.

    categorical_data : pandas.DataFrame
        DataFrame containing only the categorical features.

    Example:
    numerical_data, categorical_data = DC_numericalCategoricalSplit(my_dataframe)
    """
    numerical_features=df.select_dtypes(exclude=['object']).columns
    categorical_features=df.select_dtypes(include=['object']).columns
    numerical_data=df[numerical_features]
    categorical_data=df[categorical_features]
    return(numerical_data,categorical_data)
    
def dc_bar_plot_with_values(x_values, y_values, xlabel='', ylabel='', title=''):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    
    # Create the bar plot
    bars = ax.bar(x_values, y_values)
    
    # Add values on top of the bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.05, round(yval, 2), ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    plt.tight_layout()
    plt.show()
    
def dc_cfplot(a, b, color, title):
    """
    Plot a confusion matrix heatmap with annotations.

    Parameters:
    a : array-like, shape (n_samples,)
        True binary labels.

    b : array-like, shape (n_samples,)
        Predicted binary labels.

    color : str or Colormap, optional
        Colormap for the heatmap.

    title : str
        Title for the confusion matrix plot.

    Returns:
    None
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import numpy as np
    cf_matrix = confusion_matrix(a, b)
    group_names = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1,v2,v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.array(labels).reshape(2,2)
    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap = color)
    ax.set_title(title+"\n\n");
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values');
    plt.show()
    
def dc_plot_roc_curve(fpr, tpr, label = None, rocscore = None):
    """
    Plot the Receiver Operating Characteristic (ROC) curve.

    Parameters:
    fpr : array-like
        False Positive Rate values.

    tpr : array-like
        True Positive Rate values.

    label : str, optional
        Label for the ROC curve.

    rocscore : float, optional
        ROC AUC score to display in the plot.

    Returns:
    None
    """
    
    if rocscore is None:
        r = ""
    else:
        r = str(rocscore)
    plt.plot(fpr, tpr, linewidth = 2, label = label)
    plt.plot([0,1], [0,1], 'k--')
    plt.axis([0,1,0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(" AUC Plot ")
    plt.show()
    
def dc_nullFind(df):
    """
    Find and count missing (null) values in a DataFrame.

    Parameters:
    df : pandas.DataFrame
        The input DataFrame to check for missing values.

    Returns:
    null_numerical : pandas.Series
        Series containing counts of missing values for each numerical feature, sorted in descending order.

    null_categorical : pandas.Series
        Series containing counts of missing values for each categorical feature, sorted in descending order.

    Example:
    null_numerical, null_categorical = DC_nullFind(my_dataframe)
    """
    null_numerical=pd.isnull(df).sum().sort_values(ascending=False)
    #null_numerical=null_numerical[null_numerical>=0]
    null_categorical=pd.isnull(df).sum().sort_values(ascending=False)
    # null_categorical=null_categorical[null_categorical>=0]
    return(null_numerical,null_categorical)