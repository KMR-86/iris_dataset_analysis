import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
iris = pd.read_csv("iris.csv")

iris_setosa = iris.loc[iris["species"] == "setosa"]
iris_virginica = iris.loc[iris["species"] == "virginica"]
iris_versicolor = iris.loc[iris["species"] == "versicolor"]

def main():

    print ("iris dataset shape : ",iris.shape)
    print ("all the columns:\n",iris.columns)
    print("species distribution",iris["species"].value_counts())
    print("so it means that iris dataset is a balanced dataset")

    print("plot of sepal length and sepal width:\n")
    sns.set_style("whitegrid")
    sns.FacetGrid(iris, hue="species", height=4) \
    .map(plt.scatter, "sepal_length", "sepal_width") \
    .add_legend()
    plt.show()
    print("pair plot of iris dataset:\n")
    plt.close()
    sns.set_style("whitegrid")
    sns.pairplot(iris, hue="species", height=2)
    plt.show()
    print("from the pair plots, we can see that petal length and petal plots are most useful datasets")



    print("iris setosa full: \n",iris_setosa)

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
    main()
    

