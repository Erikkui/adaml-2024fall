import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

class PLSRegressionWrapper:
    def __init__(self, n_components=2):
        """
        Initialize PLS Regression model
        
        Parameters:
        -----------
        n_components : int
            Number of components to use in the PLS model
        """
        self.n_components = n_components
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = PLSRegression(n_components=n_components)
        
    def fit(self, X, y):
        """
        Fit the PLS model
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values
        """
        # Scale the data
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1))
        
        # Fit the model
        self.model.fit(X_scaled, y_scaled)
        return self
    
    def predict(self, X):
        """
        Make predictions using the PLS model
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
        """
        X_scaled = self.scaler_X.transform(X)
        y_pred_scaled = self.model.predict(X_scaled)
        return self.scaler_y.inverse_transform(y_pred_scaled)
    
    def score(self, X, y):
        """Calculate R² score"""
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
    
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance based on PLS components
        
        Parameters:
        -----------
        feature_names : list, optional
            List of feature names
        """
        # Calculate VIP scores
        t = self.model.x_scores_
        w = self.model.x_weights_
        q = self.model.y_loadings_
        
        p, h = w.shape
        vips = np.zeros((p,))
        
        s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
        total_s = np.sum(s)
        
        for i in range(p):
            weight = np.array([ (w[i,j] / np.sqrt(np.sum(w[:,j]**2))) * np.sqrt(s[j]) for j in range(h)])
            vips[i] = np.sqrt(p * (np.sum(weight**2)) / total_s)
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(p)]
            
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'VIP_Score': vips
        })
        return importance_df.sort_values('VIP_Score', ascending=False)

def generate_sample_data(n_samples=100, n_features=5):
    """Generate sample data for demonstration"""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = 3*X[:, 0] + 2*X[:, 1]**2 + np.sin(X[:, 2]) + np.random.randn(n_samples)*0.1
    return X, y

def plot_results(y_test, y_pred, n_components):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs Predicted Values - PLS Regression\n(n_components={n_components})')
    plt.tight_layout()
    plt.show()

def main():
    # Generate or load your data
    X, y = generate_sample_data(n_samples=200, n_features=5)
    feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train the model
    n_components = 3
    pls = PLSRegressionWrapper(n_components=n_components)
    pls.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pls.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = pls.score(X_test, y_test)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Get feature importance
    importance_df = pls.get_feature_importance(feature_names)
    print("\nFeature Importance (VIP Scores):")
    print(importance_df)
    
    # Plot results
    plot_results(y_test, y_pred, n_components)

if __name__ == "__main__":
    main()