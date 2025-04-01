import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def plot_relational_plot(df):
    """Create a relational plot showing sepal length vs width"""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x='sepal_length',
        y='sepal_width',
        hue='species',
        style='species',
        s=100
    )
    plt.title('Sepal Length vs Width by Species')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')
    plt.legend(title='Species')
    plt.savefig('relational_plot.png')
    plt.close()
    return


def plot_categorical_plot(df):
    """Create a categorical plot showing petal length distribution"""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=df,
        x='species',
        y='petal_length',
        palette='viridis'
    )
    plt.title('Petal Length Distribution by Species')
    plt.xlabel('Species')
    plt.ylabel('Petal Length (cm)')
    plt.savefig('categorical_plot.png')
    plt.close()
    return


def plot_statistical_plot(df):
    """Create a statistical plot showing feature correlations"""
    fig, ax = plt.subplots(figsize=(10, 6))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap='coolwarm',
        vmin=-1,
        vmax=1
    )
    plt.title('Feature Correlation Heatmap')
    plt.savefig('statistical_plot.png')
    plt.close()
    return


def statistical_analysis(df, col: str):
    """Calculate statistical moments for a given column"""
    data = df[col].dropna()
    mean = np.mean(data)
    stddev = np.std(data)
    skew = ss.skew(data)
    excess_kurtosis = ss.kurtosis(data)
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """Preprocess the data with basic checks"""
    print("\nData Overview:")
    print(df.head())
    
    print("\nBasic Statistics:")
    print(df.describe())
    
    print("\nCorrelation Matrix:")
    print(df.select_dtypes(include=[np.number]).corr())
    
    # No missing values in Iris dataset, but good practice to check
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    return df


def writing(moments, col):
    """Interpret and print statistical moments"""
    print(f'\nFor the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')
    
    # Interpret skewness
    if moments[2] > 0.5:
        skew_text = "right skewed"
    elif moments[2] < -0.5:
        skew_text = "left skewed"
    else:
        skew_text = "not skewed"
    
    # Interpret kurtosis
    if moments[3] > 1:
        kurt_text = "leptokurtic"
    elif moments[3] < -1:
        kurt_text = "platykurtic"
    else:
        kurt_text = "mesokurtic"
    
    print(f'The data was {skew_text} and {kurt_text}.')
    return


def perform_clustering(df, col1, col2):
    """Perform K-means clustering on selected columns"""
    
    def plot_elbow_method():
        """Plot elbow method to determine optimal clusters"""
        inertias = []
        for k in range(1, 10):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(scaled_data)
            inertias.append(kmeans.inertia_)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.plot(range(1, 10), inertias, marker='o')
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.savefig('elbow_plot.png')
        plt.close()
        return

    def one_silhouette_inertia():
        """Calculate silhouette score and inertia for optimal clusters"""
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(scaled_data)
        _score = silhouette_score(scaled_data, labels)
        _inertia = kmeans.inertia_
        return _score, _inertia

    # Gather data and scale
    data = df[[col1, col2]].values
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Find best number of clusters
    score, inertia = one_silhouette_inertia()
    print(f"\nClustering Metrics - Silhouette Score: {score:.3f}, Inertia: {inertia:.3f}")
    plot_elbow_method()
    
    # Perform final clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(scaled_data)
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    return labels, data, centers[:, 0], centers[:, 1], kmeans.labels_


def plot_clustered_data(labels, data, xkmeans, ykmeans, centre_labels):
    """Plot the clustered data with cluster centers"""
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        data[:, 0],
        data[:, 1],
        c=labels,
        cmap='viridis',
        s=100,
        alpha=0.7
    )
    ax.scatter(
        xkmeans,
        ykmeans,
        c='red',
        marker='X',
        s=200,
        label='Cluster Centers'
    )
    plt.title('K-means Clustering of Iris Dataset')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.legend()
    plt.savefig('clustering.png')
    plt.close()
    return


def perform_fitting(df, col1, col2):
    """Perform polynomial regression fitting"""
    # Gather data
    x = df[col1].values.reshape(-1, 1)
    y = df[col2].values
    
    # Fit 2nd degree polynomial model
    model = make_pipeline(
        PolynomialFeatures(degree=2),
        LinearRegression()
    )
    model.fit(x, y)
    
    # Predict across x range
    x_fit = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
    y_fit = model.predict(x_fit)
    
    return np.column_stack((x, y)), x_fit, y_fit


def plot_fitted_data(data, x, y):
    """Plot the fitted polynomial regression"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        data[:, 0],
        data[:, 1],
        c='blue',
        label='Actual Data'
    )
    ax.plot(
        x,
        y,
        c='red',
        linewidth=3,
        label='Fitted Curve'
    )
    plt.title('Polynomial Regression Fit')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.legend()
    plt.savefig('fitting.png')
    plt.close()
    return


def main():
    # Load Iris dataset
    df = sns.load_dataset('iris')
    
    # Preprocess data
    df = preprocessing(df)
    
    # Choose column for statistical analysis
    col = 'petal_length'
    
    # Generate plots
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    
    # Perform statistical analysis
    moments = statistical_analysis(df, col)
    writing(moments, col)
    
    # Perform clustering (using sepal_length and petal_length)
    clustering_results = perform_clustering(df, 'sepal_length', 'petal_length')
    plot_clustered_data(*clustering_results)
    
    # Perform fitting (using sepal_length to predict petal_length)
    fitting_results = perform_fitting(df, 'sepal_length', 'petal_length')
    plot_fitted_data(*fitting_results)
    
    print("\nAll plots saved successfully.")
    return


if __name__ == '__main__':
    main()