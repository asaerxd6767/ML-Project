import json
path_in = 'D:/vsc_codes/ML-Project/Lab02/lab2-HOUSEPRICE_LinearRegression.ipynb'
path_out = 'D:/vsc_codes/ML-Project/Lab02/team/USA_Housing_LinearRegression.ipynb'
with open(path_in) as f:
    nb = json.load(f)
cells = []
cells.append({'cell_type':'markdown','metadata':{},'source':['# USA Housing Price Prediction (Linear Regression)']})
cells.append({'cell_type':'markdown','metadata':{},'source':['#### Dataset: USA_Housing.csv']})
cells.append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':['# import required libraries\nimport pandas as pd\nimport numpy as np\nimport seaborn as sns\nimport matplotlib.pyplot as plt']})
cells.append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':['# read the csv file\nData = pd.read_csv("USA_Housing.csv")']})
cells.append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':['Data.shape']})
cells.append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':['Data.head()']})
cells.append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':['Data.info()']})
cells.append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':['Data.describe()']})
cells.append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':['Data.corr()']})
cells.append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':['plt.figure(figsize=(10,8))\nsns.heatmap(Data.corr(), annot=True)']})
cells.append({'cell_type':'markdown','metadata':{},'source':['# Exploratory Data Analysis']})
cells.append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':['sns.scatterplot(x=Data["Avg. Area Income"], y=Data["Price"])']})
cells.append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':['sns.pairplot(Data)']})
cells.append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':['sns.displot(Data["Price"])']})
cells.append({'cell_type':'markdown','metadata':{},'source':['# Simple Linear Regression (Single Feature)']})
cells.append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':['X = Data["Avg. Area Income"]\ny = Data["Price"]']})
cells.append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':['from sklearn.model_selection import train_test_split\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)']})
cells.append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':['from sklearn.linear_model import LinearRegression\nregressor = LinearRegression()\nregressor.fit(X_train.values.reshape(-1,1), y_train)']})
cells.append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':['print(regressor.intercept_)\nprint(regressor.coef_)']})
cells.append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':['y_pred = regressor.predict(X_test.values.reshape(-1,1))']})
cells.append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':['df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})\ndf.head(10)']})
cells.append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':['from sklearn import metrics\nimport numpy as np\nprint("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))\nprint("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))\nprint("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))']})
nb['cells'] = cells
with open(path_out,'w') as f:
    json.dump(nb,f,indent=2)
print('ok')
