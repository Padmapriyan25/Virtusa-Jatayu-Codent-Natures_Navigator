from flask import Flask, render_template, request
from fileinput import filename
app = Flask(__name__)
import pandas as pd
import numpy as np
from flask import Response
import matplotlib.pyplot as plt
import seaborn as sns
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import xgboost as xgb
from sklearn.metrics import mean_squared_error


# Root endpoint
@app.get('/')
def upload():
	return render_template('upload-excel.html')


@app.post('/view')
def view():
    color_pal = sns.color_palette()
    plt.style.use('fivethirtyeight')
    plt.rcParams.update({'font.size': 6})  # Increase font size
    print('hi')
    df = pd.read_csv(request.files['file'])
    df = df.set_index('Datetime')
    df.index = pd.to_datetime(df.index)

    # Create the figure and subplots with larger sizes
    fig, axs = plt.subplots(3, 2, figsize=(16, 18))

    # Plot 1: PJME Energy Use in MW
    df.plot(style='.', ax=axs[0, 0], color=color_pal[0], title='PJME Energy Use in MW')
    axs[0, 0].grid(True)  # Add gridlines

    # Train/Test split and Data Train/Test Split
    train = df.loc[df.index < '01-01-2015']
    test = df.loc[df.index >= '01-01-2015']
    train.plot(ax=axs[1, 0], label='Training Set', title='Data Train/Test Split')
    test.plot(ax=axs[1, 0], label='Test Set')
    axs[1, 0].axvline(pd.to_datetime('01-01-2015'), color='black', ls='--')
    axs[1, 0].legend(['Training Set', 'Test Set'])
    axs[1, 0].grid(True)  # Add gridlines

    # Week Of Data
    df.loc[(df.index > '01-01-2011') & (df.index < '01-08-2011')].plot(ax=axs[2, 0], title='Week Of Data')
    axs[2, 0].grid(True)  # Add gridlines

    # Create time series features based on time series index
    def create_features(df):
        df = df.copy()
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['quarter'] = df.index.quarter
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['dayofyear'] = df.index.dayofyear
        df['dayofmonth'] = df.index.day
        df['weekofyear'] = df.index.isocalendar().week
        return df

    # Train/Test split and XGBoost modeling
    train = create_features(train)
    test = create_features(test)
    FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']
    TARGET = 'PJME_MW'
    X_train = train[FEATURES]
    y_train = train[TARGET]
    X_test = test[FEATURES]
    y_test = test[TARGET]

    reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                        n_estimators=1000,
                        early_stopping_rounds=50,
                        objective='reg:linear',
                        max_depth=3,
                        learning_rate=0.01)
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=100)

    # Plot feature importance
    fi = pd.DataFrame(data=reg.feature_importances_, index=FEATURES, columns=['importance'])
    fi.sort_values('importance').plot(kind='barh', title='Feature Importance', ax=axs[0, 1])
    axs[0, 1].grid(True)  # Add gridlines

    # Raw Data and Predictions
    test['prediction'] = reg.predict(X_test)
    df = df.merge(test[['prediction']], how='left', left_index=True, right_index=True)
    ax = df[['PJME_MW']].plot(ax=axs[1, 1], figsize=(16, 6))
    df['prediction'].plot(ax=ax, style='.')
    ax.legend(['Truth Data', 'Predictions'])
    ax.set_title('Raw Data and Prediction')
    ax.grid(True)  # Add gridlines

    # Week Of Data - Truth Data and Prediction
    df.loc[(df.index > '04-01-2018') & (df.index < '04-08-2018')]['PJME_MW'].plot(ax=axs[2, 1], figsize=(16, 6), title='Week Of Data')
    df.loc[(df.index > '04-01-2018') & (df.index < '04-08-2018')]['prediction'].plot(ax=axs[2, 1], style='.')
    axs[2, 1].legend(['Truth Data', 'Prediction'])
    axs[2, 1].grid(True)  # Add gridlines

    # Calculate and print RMSE score
    score = np.sqrt(mean_squared_error(test['PJME_MW'], test['prediction']))
    print(f'RMSE Score on Test set: {score:0.2f}')

    # Calculate and display top 10 error dates
    test['error'] = np.abs(test[TARGET] - test['prediction'])
    test['date'] = test.index.date
    top_10_errors = test.groupby(['date'])['error'].mean().sort_values(ascending=False).head(10)
    print(top_10_errors)

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the plot
    print('end')
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')
if __name__ == '__main__':
 app.run()