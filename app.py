from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

app = Flask(_name_)
app.secret_key = 'your_secret_key'

df = pd.read_csv('real_estate_tamilnadu.csv')

users = {'user1': 'user123'}

def preprocess_data(df):
    label_encoder = LabelEncoder()
    df['Location'] = label_encoder.fit_transform(df['Location'])
    df = df.dropna()
    return df

df = preprocess_data(df)

X = df[['Square_Feet', 'Bedrooms', 'Location', 'Age_of_Property']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor()
model.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    heatmap_chart = generate_heatmap()
    location_price_chart = generate_location_price_chart()
    lr_mse, rf_mse, xgb_mse, lr_mae, rf_mae, xgb_mae, lr_r2, rf_r2, xgb_r2 = compare_models()
    xgb_chart = create_xgb_chart()
    lr_chart = create_linear_regression_chart()
    rf_chart = create_rf_chart()

    return render_template('about.html', heatmap_chart=heatmap_chart,
                           location_price_chart=location_price_chart,
                           lr_mse=lr_mse, rf_mse=rf_mse, xgb_mse=xgb_mse,
                           lr_mae=lr_mae, rf_mae=rf_mae, xgb_mae=xgb_mae,
                           lr_r2=lr_r2, rf_r2=rf_r2, xgb_r2=xgb_r2,
                           xgb_chart=xgb_chart, lr_chart=lr_chart, rf_chart=rf_chart)

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users and users[username] == password:
            return redirect(url_for('user_dashboard'))
        else:
            flash('Invalid username or password', 'error')

    return render_template('login.html')

@app.route('/user_dashboard', methods=['GET', 'POST'])
def user_dashboard():
    if request.method == 'POST':
        try:
            square_feet = int(request.form['Square_Feet'])
            bedrooms = int(request.form['Bedrooms'])
            area = request.form['Location']
            age_of_property = int(request.form['Age_of_Property'])

            if area == '':
                flash("Please enter a location.", 'error')
                return redirect(url_for('user_dashboard'))

            label_encoder = LabelEncoder()
            area_encoded = label_encoder.fit_transform([area])[0]

            input_data = [[square_feet, bedrooms, area_encoded, age_of_property]]
            predicted_price = model.predict(input_data)[0]

            market_trends_chart = create_market_trends_chart()
            price_prediction_chart = create_price_prediction_chart(input_data, predicted_price, square_feet)
            forecast_chart = create_price_forecasting_chart()

            return render_template('user_dashboard.html',
                                   predicted_price=predicted_price,
                                   market_trends_chart=market_trends_chart,
                                   price_prediction_chart=price_prediction_chart,
                                   forecast_chart=forecast_chart)

        except KeyError as e:
            flash(f"Missing form field: {str(e)}", 'error')
            return redirect(url_for('user_dashboard'))
        except ValueError as e:
            flash("Please enter valid data in all fields.", 'error')
            return redirect(url_for('user_dashboard'))

    return render_template('user_dashboard.html')

def generate_heatmap():
    correlation_matrix = df.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

def generate_location_price_chart():
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Location', y='Price', data=df, palette='coolwarm')
    plt.title('Price Distribution by Location')
    plt.xlabel('Location')
    plt.ylabel('Price')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

def create_market_trends_chart():
    plt.figure(figsize=(10, 6))
    trends_data = df.groupby('Location')['Price'].mean().sort_values()
    trends_data.plot(kind='barh', color='skyblue')
    plt.title('Market Trends by Area')
    plt.xlabel('Average Price')
    plt.ylabel('Area')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

def create_price_prediction_chart(input_data, predicted_price, square_feet):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Square_Feet'], df['Price'], color='blue', label='Dataset')
    plt.scatter(square_feet, predicted_price, color='red', label=f'Predicted Price: {predicted_price}')
    plt.title('Price Prediction vs Square Feet')
    plt.xlabel('Square Feet')
    plt.ylabel('Price')
    plt.legend()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

def create_price_forecasting_chart():
    plt.figure(figsize=(10, 6))
    sizes = range(1000, 5000, 500)
    predicted_prices = model.predict([[size, 3, 1, 20] for size in sizes])
    plt.plot(sizes, predicted_prices, marker='o', color='green', label='Price Forecast')
    plt.title('Real Estate Price Forecasting')
    plt.xlabel('Property Size (Square Feet)')
    plt.ylabel('Predicted Price')
    plt.legend()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

def create_xgb_chart():
    model_xgb = XGBRegressor()
    model_xgb.fit(X_train, y_train)
    y_pred_xgb = model_xgb.predict(X_test)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_xgb, color='orange', label='Predicted vs Actual')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linewidth=2, label='Perfect Prediction')
    plt.title('XGBoost: Predicted vs Actual Prices')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.legend()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

def create_linear_regression_chart():
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    y_pred_lr = model_lr.predict(X_test)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_lr, color='blue', label='Predicted vs Actual')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linewidth=2, label='Perfect Prediction')
    plt.title('Linear Regression: Predicted vs Actual Prices')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.legend()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

def create_rf_chart():
    model_rf = RandomForestRegressor()
    model_rf.fit(X_train, y_train)
    y_pred_rf = model_rf.predict(X_test)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_rf, color='green', label='Predicted vs Actual')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linewidth=2, label='Perfect Prediction')
    plt.title('Random Forest: Predicted vs Actual Prices')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.legend()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

def compare_models():
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(),
        'XGBoost': XGBRegressor()
    }
    mse_dict, mae_dict, r2_dict = {}, {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse_dict[name] = mean_squared_error(y_test, y_pred)
        mae_dict[name] = mean_absolute_error(y_test, y_pred)
        r2_dict[name] = r2_score(y_test, y_pred)
    return mse_dict['Linear Regression'], mse_dict['Random Forest'], mse_dict['XGBoost'], \
           mae_dict['Linear Regression'], mae_dict['Random Forest'], mae_dict['XGBoost'], \
           r2_dict['Linear Regression'], r2_dict['Random Forest'], r2_dict['XGBoost']

if _name_ == '_main_':
    app.run(debug=True) 
