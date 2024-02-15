import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, classification_report
import pandas as pd

data = pd.read_csv('heart_patient_fixed_11.csv', sep=',')

data.info()

#Graphics
sns.pairplot(data, corner=True)
plt.savefig('pairplot')

x = ['age', 'thalach', 'trestbps', 'chol']
c = 'restecg'
y = ['thalach', 'trestbps']

sns.pairplot(data, x_vars=x, y_vars=y)
plt.savefig('pairplot_xy')

data.hist('age')
plt.savefig('agehist')

sns.displot(data=data, x='chol', kind='kde') # equal sns.kdeplot(data=data, x='chol')
plt.show()

sns.displot(data=data, x='chol', hue='sex', kind='kde') # equal sns.kdeplot(data=data, x='chol', hue='sex')
plt.savefig('cholsexkde')

jp = sns.jointplot(data=data, x='chol', y='age', kind='reg', marker='*', marginal_ticks=True)
jp.plot_joint(sns.kdeplot, color='g')
plt.savefig('cholagejp')

sns.heatmap(data[['age', 'thalach', 'trestbps', 'chol']].corr(),
            annot=True, vmin=0, vmax=1, linewidth=5, cmap='spring')
plt.savefig('heatmap')

cp = sns.catplot(data=data, x='sex', y=x[1], col=c, kind='box')
cp.set_axis_labels("", x[1])
cp.set_xticklabels(['Female', 'Male'])
cp.despine(left=True)
plt.savefig('thalachbox')

cp = sns.catplot(data=data, x='sex', y=x[2], col=c, kind='box')
cp.set_axis_labels("", x[2])
cp.set_xticklabels(['Female', 'Male'])
cp.despine(left=True)
plt.savefig('trestbpsbox')

cp = sns.catplot(data=data, x='sex', y=x[3], col=c, kind='box')
cp.set_axis_labels("", x[3])
cp.set_xticklabels(['Female', 'Male'])
cp.despine(left=True)
plt.savefig('cholbox')

cp = sns.catplot(data=data, x='sex', y=x[1], col=c, kind='violin')
cp.set_axis_labels("", x[3])
cp.set_xticklabels(['Female', 'Male'])
cp.despine(left=True)
plt.savefig('thalchviolin')

#Classificator

X = data[['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach']]
y = data['num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(f'TRAIN: {X_train.shape}, {y_train.shape} TEST: {X_test.shape},{y_test.shape}')

treeModel = tree.DecisionTreeClassifier(max_depth=5)
treeModel.fit(X_train, y_train)

featuresNames = X.columns
print(featuresNames)


plt.figure(figsize=(20, 15))
tree.plot_tree(treeModel, feature_names=featuresNames, filled=True)
# plt.show()
plt.savefig('treeHeart_patient_fixed.png')

y_prediction = treeModel.predict(X_test)

print(f'Average absolute error: {round(mean_absolute_error(y_test, y_prediction), 2)} Average squared error: {round(mean_squared_error(y_test, y_prediction), 2)}')

print(confusion_matrix(y_true=y_test, y_pred=y_prediction))
print(classification_report(y_true=y_test, y_pred=y_prediction))