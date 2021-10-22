# Extra Day 14

## [Feature Engineering](https://www.kaggle.com/learn/feature-engineering)

- Tutorial: The link of Kaggle tutorial.
- Course Content: Almost the same content from Tutorial, maybe I added a little change. You can run it on your local machine.
- Exercise: My exercise answer from Kaggle course exercise. You cannot run it on your local machine.

---

### What is Feature Engineering

> [Tutorial](https://www.kaggle.com/ryanholbrook/what-is-feature-engineering)<br>
> [Course Content](concrete-baseline.ipynb)

Packages:
- [sklearn.model_selection.cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)

---

### Mutual Information

> [Tutorial](https://www.kaggle.com/ryanholbrook/mutual-information)<br>
> [Course Content](automobile-mutual-information.ipynb)<br>
> [Exercise](exercise-mutual-information.ipynb)

Packages:
- [sklearn.feature_selection.mutual_info_regression](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html)<br>
- [sklearn.feature_selection.mutual_info_classif](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html)

---

### Creating Features

> [Tutorial](https://www.kaggle.com/ryanholbrook/creating-features)<br>
> [Course Content](creating-features.ipynb)<br>
> [Exercise](exercise-creating-features.ipynb)

---

### Clustering with K-Means

> [Tutorial](https://www.kaggle.com/ryanholbrook/clustering-with-k-means)<br>
> [Course Content](clustering-with-k-means.ipynb)<br>
> [Exercise](exercise-clustering-with-k-means.ipynb)

Some features are meaningless for ML. For example, longitude and latitude are just numbers, and they have unlimited combinations. However, if we can identify their clusters, the *area (longitude & latitude)* is meaningful for prediction.

Packages:
- [sklearn.cluster.KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)

---

### Principal Component Analysis

> [Tutorial](https://www.kaggle.com/ryanholbrook/principal-component-analysis)<br>
> [Course Content](principal-component-analysis.ipynb)<br>
> [Exercise](exercise-principal-component-analysis.ipynb)

Packages:
- [sklearn.decomposition.PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

---


### Target Encoding

> [Tutorial](https://www.kaggle.com/ryanholbrook/target-encoding)<br>
> [Course Content](target-encoding.ipynb)<br>
> [Exercise](exercise-target-encoding.ipynb)

Packages:
- [category_encoders](https://contrib.scikit-learn.org/category_encoders/)