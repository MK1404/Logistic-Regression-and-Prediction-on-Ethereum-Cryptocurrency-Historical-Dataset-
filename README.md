# Logistic-Regression-and-Prediction-on-Ethereum-Cryptocurrency-Historical-Dataset-
analysis and prediction on ethereum cryptocurrancy dataset  
through logistic regression model 
here I used standard scaler to standardized the dataset for better accuracy

#importing essential libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#loading the dataset
df=pd.read_csv("transaction_dataset.csv")
df

df.columns 

df.info()  

#drawing a heatmap to check null values in dataset
plt.figure(figsize=(12,6))
sns.heatmap(df.isnull(), cbar=False)
plt.show()

#creating a function to drop the null values 
def clean_data(df):                    #you may use df.dropna to drop null values from whole dataset
    df.dropna(inplace=True)
    print(df.isnull().sum())

clean_data(df)

df.drop(["Unnamed: 0","Address"],axis=1,inplace=True) #removing unwanted columns from datset

df

#preprocessing this will convert str to int 
from sklearn import preprocessing
encoder = preprocessing.LabelEncoder()
for i in df.columns:
    if isinstance(df[i][0], str):
            df[i] = encoder.fit_transform(df[i])

df.tail()

df.drop(["Index"],axis=1,inplace=True)

df

sns.countplot(df.FLAG)             

print(df['FLAG'].value_counts())

labels = ['Non-fraud', 'Fraud']
colors = ['#f9ae35', '#f64e38']
plt.pie(x = df['FLAG'].value_counts(), autopct='%.2f%%' , explode=[0.02]*2, labels=labels, pctdistance=0.5, textprops={'fontsize': 14}, colors = colors)
plt.title('Target distribution')
plt.show()

df

li = df[df['FLAG']==0].sample(6260).index #sampling for flag columns to equal the values in 0,1

df = df.drop(li, axis = 0)
df.FLAG.value_counts()

X=df.drop("FLAG",axis=1)
y=df["FLAG"]                    #define  output column to y

#standardized the dataset to get  better result
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaled_df=scaler.fit_transform(X)

#crreating a dataframe with standardized values of x
scale_df = pd.DataFrame(scaled_df)
scale_df
scale_df.columns=X.columns
scale_df.head()

x=scale_df

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,stratify=y)

print(f"Rows in train_set: {len(x_train)}\nRows in test_set: {len(x_test)}\n")

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

# Train the model using 'fit' method
model.fit(x_train, y_train)

# Test the model using 'predict' method
y_pred = model.predict(x_test)

# Print the classification report 
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


#printing a confusion matrix for y_test,y_pred
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print('Print the Confusion Matrix')
print(cm)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu");

# The End
