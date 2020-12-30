import pandas
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as pltimg



df = pandas.read_csv("shows2.csv")

d = {'soleil': 0, 'couvert': 1, 'pluie': 2 }
df['Temps'] = df['Temps'].map(d)

d = {'chaude': 0, 'bonne': 1, 'fraîche': 2}
df['température'] = df['température'].map(d)

d = {'haute': 1, 'normale': 0}
df['humidité'] = df['humidité'].map(d)

print(df)

features = ['Temps', 'température', 'humidité', 'vent' ]

X = df[features]
y = df['Football']

print(X)
print(y)

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)
data = tree.export_graphviz(dtree, out_file=None, feature_names=features)
graph = pydotplus.graph_from_dot_data(data)
graph.write_png('mydecisiontree.png')

img=pltimg.imread('mydecisiontree.png')
#plt.savefig('mydecisiontree.png')
imgplot = plt.imshow(img)
plt.show()

 
