# Day 66

## [Data Cleaning](https://www.kaggle.com/learn/data-cleaning)

- Tutorial: The link of Kaggle tutorial.
- Course Content: Almost the same content from Tutorial, maybe I added a little change. You can run it on your local machine.
- Exercise: My exercise answer from Kaggle course exercise. You cannot run it on your local machine.

> [Data Source](https://www.kaggle.com/maxhorowitz/nflplaybyplay2009to2016)

---

### Handling Missing Values

> [Tutorial](https://www.kaggle.com/alexisbcook/handling-missing-values)<br>
> [Exercise](exercise-handling-missing-values.ipynb)

-  Is this value missing because it wasn't recorded or because it doesn't exist?

> [Imputation of missing values](https://scikit-learn.org/stable/modules/impute.html)

Packages:
- [pandas.DataFrame.fillna](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html)<br>
- [sklearn.impute.SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)

---

### Scaling and Normalization

> [Tutorial](https://www.kaggle.com/alexisbcook/scaling-and-normalization)<br>
> [Exercise](exercise-scaling-and-normalization.ipynb)

Advanced Content in [Exercise](exercise-scaling-and-normalization.ipynb).

Packages
- [sklearn.preprocessing.MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
- [sklearn.preprocessing.StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)<br>
- [sklearn.preprocessing.Normalizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html)<br>

---

### Parsing Dates

> [Tutorial](https://www.kaggle.com/alexisbcook/parsing-dates)<br>
> [Exercise](exercise-parsing-dates.ipynb)

Advanced Content in [Exercise](exercise-parsing-dates.ipynb).

Packages:
- [pandas.to_datetime](https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html)<br>
- [pandas.Series.dt](https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.html)

---

### Character Encodings

> [Tutorial](https://www.kaggle.com/alexisbcook/character-encodings)<br>
> [Exercise](exercise-character-encodings.ipynb)

Advanced Content in [Exercise](exercise-character-encodings.ipynb).

Packages:
- [chardet](https://pypi.org/project/chardet/): The Universal Character Encoding Detector

---

### Inconsistent Data Entry

> [Tutorial](https://www.kaggle.com/alexisbcook/inconsistent-data-entry)<br>
> [Exercise](exercise-inconsistent-data-entry.ipynb)

Packages:
- [fuzzywuzzy](https://pypi.org/project/fuzzywuzzy/): Fuzzy string matching like a boss. It uses [Levenshtein Distance](https://en.wikipedia.org/wiki/Levenshtein_distance) to calculate the differences between sequences in a simple-to-use package.
