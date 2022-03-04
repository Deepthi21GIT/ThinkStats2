import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pygments.lexers import go
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import statistics
from scipy import stats
from empiricaldist import Pmf, empiricaldist

from thinkplot import pmf
from thinkstats2 import Pmf

meddata = pd.read_csv("../code/insurance.csv")
meddata.head()
meddata.info()
meddata.describe()

sns.barplot(x = 'sex', y ='charges', data = meddata)
sns.boxplot(x= 'sex', y='charges', hue='smoker', data = meddata)
sns.displot(meddata['charges'], bins = 50)
sns.pairplot(meddata)
sns.jointplot(x = 'age', y = 'charges', data = meddata)


fgmap = sns.FacetGrid(meddata, col ='age')
fgmap.map(plt.hist, 'age')

fgmap = sns.FacetGrid(meddata, col ='sex')
fgmap.map(plt.hist, 'sex')

fgmap = sns.FacetGrid(meddata, col ='children')
fgmap.map(plt.hist, 'children')

fgmap = sns.FacetGrid(meddata, col ='smoker')
fgmap.map(plt.hist, 'smoker')

fgmap = sns.FacetGrid(meddata, col ='region')
fgmap.map(plt.hist, 'region')

age_des= meddata['age'].describe()
age_des['mean']
child_des= meddata['children'].describe()
child_des['mean']
bmi_des= meddata['bmi'].describe()
bmi_des['mean']
charges_des= meddata['charges'].describe()
charges_des['mean']

plt.figure(figsize= (12,7))
sns.displot(meddata['age'])
plt.title('Age distribution')
ax = sns.displot(data = meddata, x = 'age', kde = True)
sns.countplot(meddata['age'])
plt.axvline(39, linestyle = '--', color = 'green', label = 'mean Age')
plt.axvline(age_des['mean'], linestyle = "--", color = "red")
plt.title('Age countplot Distribution')

meddata['sex'].value_counts()
sns.countplot(meddata['sex'])
plt.title("Gender of the peoples")


ax= plt.figure(figsize=(15,6))
sns.countplot(meddata['bmi'])
sns.displot(meddata[meddata.bmi <= 30]['charges'],color = 'r')
plt.title("Charges for people who are below BMI 30")
plt.axvline(bmi_des['mean'], linestyle = "--", color = "red")
ax= plt.figure(figsize=(15,6))
sns.countplot(meddata['bmi'])
sns.displot(meddata[meddata.bmi >= 30]['charges'],color = 'r')
plt.title("Charges for people who are above BMI 30")
plt.title('BMI distribution')
sns.jointplot(x='bmi',y='charges' ,data = meddata);
sns.jointplot(x='bmi',y='charges',hue ='smoker' ,data = meddata);
sns.lmplot(x= 'bmi',y='charges',hue='smoker',data = meddata)

ax= plt.figure(figsize=(20,6))
sns.countplot(meddata['children'])
sns.displot( meddata['children'])
sns.lmplot(x= 'children',y='charges',hue='smoker',data = meddata)
plt.axvline(child_des['mean'], linestyle = "--", color = "red", )
plt.title("Distribution of Children")

ax= plt.figure(figsize=(15,6))
sns.countplot(meddata['smoker'])
plt.figure(figsize= (12,7))
sns.boxplot(y= 'smoker',x = 'charges',hue='sex',orient= 'h', data = meddata)
plt.title("Comparing smoking vs non-smoking class for male and female");
sns.lmplot(x="age", y="charges", hue="smoker", data=meddata, palette = 'inferno_r', size = 7)
plt.title('Smokers and non-smokers')
keys = meddata['smoker'].value_counts().keys().to_list()
values = meddata['smoker'].value_counts().to_list()


ax= plt.figure(figsize=(6,6))
sns.countplot(meddata['charges'])
sns.displot(meddata['charges'])
plt.axvline(charges_des['mean'], linestyle = '--', color = "red")
plt.show()

plt.figure(figsize = (10,7))
sns.barplot(x='region',y='charges',data= meddata,palette='cool')
plt.figure(figsize = (7,5))
sns.countplot(x='region',hue='smoker',data= meddata)
plt.figure(figsize = (10,7))
sns.barplot(x='region',y='charges',hue='smoker',data= meddata)
plt.figure(figsize=(10, 7))
sns.barplot(x='region', y='charges', hue='children', data=meddata, palette='cool')
plt.show()

# Mean
age= statistics.mean(age_des)
print('age mean:', age)

bmi= statistics.mean(bmi_des)
print('bmi mean:', bmi)

children= statistics.mean(child_des)
print('children mean:', children)

charges= statistics.mean(charges_des)
print('charges mean:', charges)

# Median
age= statistics.median(age_des)
print('age median:', age)

bmi= statistics.median(bmi_des)
print('bmi median:', bmi)

children= statistics.median(child_des)
print('children median:', children)

charges= statistics.median(charges_des)
print('charges median:', charges)

# PMF
pmf_age = Pmf(age_des)
print(pmf_age)

# CDF
agecol = meddata['age'] / 100
chargescol = meddata['charges'] / 100

meddatalength = agecol - chargescol
print("meddatalength", meddatalength)

plt.hist(agecol, bins=20, histtype = 'step')

# Label the axes
plt.xlabel('Age Columns')
plt.ylabel('Charges incurred')

# Show the figure
plt.show()
_ = plt.plot(agecol, chargescol, marker='.', linestyle='none')
_ = plt.xlabel('Diffrence between two columns')
_ = plt.ylabel('CDF')

# Show the plot
plt.show()



from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(handle_unknown = 'ignore', dtype = int)
regions = [['southeast',0],['southwest',1],['northeast',2],['northwest',3]]
one.fit(regions)
print(one.transform(regions).toarray())


def regions_meddata(r):
    if r == 'southeast':
        return(one.transform(regions).toarray()[0].tolist())
    if r == 'southwest':
        return(one.transform(regions).toarray()[1].tolist())
    if r == 'northeast':
        return(one.transform(regions).toarray()[2].tolist())
    if r == 'northwest':
        return(one.transform(regions).toarray()[3].tolist())
    else:
        return([0])

meddata['regionsohe'] = meddata['region'].apply(regions_meddata)

meddata.head()

meddata.drop('region', axis = 1, inplace = True)
meddata.head()

region2 = pd.DataFrame(meddata)
region2[['reg0','reg1','reg2','reg3','reg4','reg5','reg6','reg7']] = pd.DataFrame(region2.regionsohe.tolist(),index = region2.index)
region2.drop('regionsohe',axis =1, inplace = True)
region2.head()

meddata = pd.DataFrame(region2)
meddata.head()

def binary_val(r):
    if (r=='female'):
        return 1
    if (r == 'yes'):
        return 1
    else:
        return 0
meddata['sex']= meddata['sex'].apply(binary_val)
meddata['smoker'] = meddata['smoker'].apply(binary_val)

meddata.head()

X = meddata.drop('charges',axis =1)
y = meddata['charges']

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X_train, X_test , y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)

X_train.head()

y_train.head()

model2 = tf.keras.Sequential([
    tf.keras.Input(shape=(13,)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(1, activation=None)
])

model2.compile(loss=tf.keras.losses.mae,
               optimizer=tf.keras.optimizers.Adam(),
               metrics=['mae']

               )

model2.fit(X_train, y_train, epochs=100)

model2.evaluate(X_train,y_train)
model2.evaluate(X_test,y_test)


plot_model(model2,show_shapes = True)
model2.summary()

y_preds = model2.predict(X_test)

tf.metrics.mean_absolute_error(y_true = y_test, y_pred= y_preds)

heatmap= meddata.corr()
sns.heatmap(meddata.corr(), annot = True)

scaler= StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(meddata), columns=meddata.columns)
scaled = scaler.fit_transform(meddata)
print(scaled)

x=meddata[['age','bmi','children']]
y=meddata['charges']

x1= sm.add_constant(x)
results = sm.OLS(y,x1)

model=results.fit()
model.summary()