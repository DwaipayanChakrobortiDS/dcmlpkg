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




def dc_UVA_numeric(data, var_group):
    '''
    Perform Univariate Analysis on Numeric Variables.

    Parameters:
    - data (DataFrame): The dataset containing the variables for analysis.
    - var_group (list): A list of variable names (strings) to analyze.

    Returns:
    - None

    This function takes a group of numeric variables (INTEGER and FLOAT) and plots the Kernel Density Estimation (KDE) along with various descriptive statistics.

    It iterates over each variable in the provided list, calculates descriptive statistics including minimum, maximum, range, mean, median, standard deviation, skewness, and kurtosis. Then, it plots the KDE plot along with markers for minimum, maximum, mean, and median.

    If a variable is not found in the dataset, it prints a warning message.

    Example:
    dc_UVA_numeric(data=my_data, var_group=['age', 'income'])
    '''
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not var_group:
        print("No variables provided for analysis.")
        return
    
    plt.figure(figsize=(7 * len(var_group), 3), dpi=100)
    
    for j, i in enumerate(var_group):
        if i not in data.columns:
            print(f"Variable '{i}' not found in the dataset.")
            continue
        
        # calculating descriptives of variables
        mini = data[i].min()
        maxi = data[i].max()
        ran = maxi - mini
        mean = data[i].mean()
        median = data[i].median()
        st_dev = data[i].std()
        skew = data[i].skew()
        kurt = data[i].kurtosis()

        # calculating the point of standard deviation 
        points = mean - st_dev, mean + st_dev

        # plotting the variables with every information
        plt.subplot(1, len(var_group), j + 1)
        sns.kdeplot(data[i], shade=True, color='LightGreen')
        sns.lineplot(points, [0, 0], color='black', label="std_dev")
        sns.scatterplot([mini, maxi], [0, 0], color='orange', label="min/max")
        sns.scatterplot([mean], [0], color='red', label="mean")
        sns.scatterplot([median], [0], color='blue', label="median")
        plt.xlabel('{}'.format(i), fontsize=20)
        plt.ylabel('density')
        plt.title("std_dev = {}; kurtosis = {};\nskew = {}; range = {}\nmean = {}; median = {}".format(
            (round(points[0], 2), round(points[1], 2)),
            round(kurt, 2),
            round(skew, 2),
            (round(mini, 2), round(maxi, 2), round(ran, 2)),
            round(mean, 2),
            round(median, 2)))
    
    plt.tight_layout()
    plt.show()


  



def dc_UVA_category(data, var_group):
    '''
    Perform Univariate Analysis on Categorical Variables.

    Parameters:
    - data (DataFrame): The dataset containing the variables for analysis.
    - var_group (list): A list of variable names (strings) to analyze.

    Returns:
    - None

    This function takes a group of categorical variables and plots the value counts along with a bar plot.

    It iterates over each variable in the provided list, calculates the normalized value counts and the number of unique categories. Then, it plots a count plot with each category's value counts annotated.

    If a variable is not found in the dataset, it prints a warning message.

    Example:
    dc_UVA_category(data=my_data, var_group=['gender', 'education'])
    '''
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not var_group:
        print("No variables provided for analysis.")
        return
    
    # setting figure size
    size = len(var_group)
    plt.figure(figsize=(8 * size, 7), dpi=100)
    
    # for every variable
    for j, i in enumerate(var_group):
        if i not in data.columns:
            print(f"Variable '{i}' not found in the dataset.")
            continue
        
        norm_count = data[i].value_counts(normalize=True)
        n_uni = data[i].nunique()

        # plotting the variable with every information
        plt.subplot(1, size, j + 1)
        graph2 = sns.countplot(y=i, data=data, order=data[i].value_counts().index, palette="Set2")
        for p in graph2.patches:
            graph2.annotate(s='{:.0f}'.format(p.get_width()), xy=(p.get_width() + 0.1, p.get_y() + 0.7))
        plt.xlabel('fraction/percent', fontsize=20)
        plt.ylabel('{}'.format(i), fontsize=20)
        plt.title('n_uniques = {}\n value counts \n {}'.format(n_uni, norm_count), fontsize=12)
    
    plt.tight_layout()
    plt.show()

