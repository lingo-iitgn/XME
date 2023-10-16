# from rich.pretty import pprint
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rich.pretty import pprint

with open('in.txt') as f:
    content = f.readlines()

content = [x.strip() for x in content]
data = []


for row in content:
    def mp(x):
        return float(x.strip())
    data.append(list(map(mp, row.split("\t"))))


# b_data = data[:6]
x_data = data[:]
# m_data = data[6:12]
row_name = ["en", "hi", "es", "fr", "bn", "gu", "mix", "inv"]
col_name = row_name[:-2]

for i in range(len(x_data)):
    for j in range(len(x_data[i])):
        x_data[i][j] = round(x_data[i][j]*100, 4)

pprint(x_data)
l = []
for i in range(len(x_data)):
    d = {"evl": row_name[i]}   
    for j in range(len(x_data[i])):
        d[col_name[j]] = x_data[i][j]
    l.append(d)

df = pd.DataFrame(l)
# pprint(df)

# Create heatmap with seaborn library
sns.heatmap(df.iloc[:, 1:], xticklabels=col_name, yticklabels=row_name, annot=True, cmap="YlGnBu", fmt=".2f")

# Save sns figure as pdf file
plt.savefig("out.pdf")

