import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold


class Prediction1:
    
    def __init__(self, data):
        self.df = data

    def prediction1_using_regression(self):
        

        # Feature selection
        features = ['grade_span_min', 'grade_span_max', 'extracurricular_activities']

        # Train-Test Split
        X = self.df[features]
        y = self.df['total_students']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Data Preprocessing
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Model Training
        model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', LinearRegression())])
        model.fit(X_train, y_train)

        # Model Evaluation
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f'Mean Squared Error: {mse}')
        print(f'R-squared: {r2}')

        # Regression Plot
        plt.figure(figsize=(10, 6))
        sns.regplot(x=y_test, y=y_pred, scatter_kws={'s': 20, 'alpha': 0.5})
        plt.xlabel('Actual Total Students')
        plt.ylabel('Predicted Total Students')
        plt.title('Regression Plot: Actual vs Predicted Total Students')
        plt.show()
        
    def prediction1_using_crossvalidation(self):
        features = ['grade_span_min', 'grade_span_max', 'school_type', 'num_ext_act']
        X = self.df[features]
        y = self.df['total_students']
        
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor())
        ])

        # Use cross-validation with 5 folds
        cv_scores = cross_val_score(model_pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
        print(f'Cross-Validation Scores: {cv_scores}')
        print(f'Mean Cross-Validation Score: {cv_scores.mean()}')

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)

        for i, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model_pipeline.fit(X_train, y_train)
            y_pred = model_pipeline.predict(X_test)

            axes[i].scatter(y_test, y_pred, alpha=0.5)
            axes[i].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--k')
            axes[i].set_title(f'Fold {i + 1}')

        plt.suptitle('Actual vs Predicted Total Students (Cross-Validation)')
        return plt.show()

    def prediction1_using_clustering(self):
        
        features = ['grade_span_min', 'grade_span_max', 'num_ext_act']
        X = self.df[features]
    
        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        kmeans = KMeans(n_clusters=3 , random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        
        plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', edgecolors='k')
        plt.title('Clustering of Schools')
        plt.xlabel('Feature 1 (Scaled)')
        plt.ylabel('Feature 2 (Scaled)')
        plt.show()
    
    def prediction2(self):
        # Extract relevant features
        features = self.df[['num_ext_act', 'num_sports_boys', 'num_sports_girls', 'num_sports_coed']]

        # Encode categorical features (e.g., one-hot encoding for sports types)
        features = pd.get_dummies(features, columns=['num_sports_boys', 'num_sports_girls', 'num_sports_coed'], drop_first=True)
        
        # Assuming your target variable is school performance, you'll need this
        target = self.df['number_programs']  # Replace 'school_performance' with the actual target column name

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        model_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor()),
        ])
        print(model_pipeline)
        
        model_pipeline.fit(X_train, y_train)

        # Make predictions on the test set
        predictions = model_pipeline.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, predictions)
        
        print(f'Mean Squared Error: {mse}')
        
        param_grid = {
            'regressor__n_estimators': [50, 100, 200],
            'regressor__max_depth': [None, 10, 20],
        }

        grid_search = GridSearchCV(model_pipeline, param_grid, cv=5)
        grid_search.fit(X_train, y_train)

        # Get the best parameters
        best_params = grid_search.best_params_
        print(f'Best Hyperparameters: {best_params}')

        # Use the best model for predictions
        best_model = grid_search.best_estimator_
        best_predictions = best_model.predict(X_test)

        # Evaluate the best model
        best_mse = mean_squared_error(y_test, best_predictions)
        print(f'Best Model Mean Squared Error: {best_mse}')
        
        # Make predictions using the best model
        predictions = best_model.predict(X_test)

        # Create a scatter plot comparing actual vs predicted values
        plt.figure(figsize=(8, 6))
        sns.regplot(x=y_test, y=predictions)
        plt.title('Actual vs Predicted Values')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        return plt.show()

    
    def prediction3(self):
        
        # Feature selection
        features = ['extracurricular_activities', 'psal_sports_boys', 'psal_sports_girls', 'number_programs']

        # Drop rows with missing values
        df = self.df[features].dropna()

        # Preprocessing
        numeric_features = self.df.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = self.df.select_dtypes(include=['object']).columns
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Apply preprocessing
        X = preprocessor.fit_transform(self.df)

        # K-means clustering
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)  # You can adjust the number of clusters
        self.df['cluster'] = kmeans.fit_predict(X)
        
        # Visualize clusters
        sns.pairplot(self.df, hue='cluster', palette='Dark2')
        plt.tight_layout()
        return plt.show()