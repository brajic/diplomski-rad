from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from pymfe.mfe import MFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Izracunaj ARR vrijednosti za svaki skup podataka
def compute_arr_values_for_dataset(X_train, y_train, algoritmi, alpha=0.05):
    results = []
    # Izracunaj preciznost i vrijeme izvršavanja za svaki algoritam
    for name, clf in algoritmi.items():
        scores = cross_validate(clf, X_train, y_train, scoring=['accuracy'], cv=10, return_train_score=False, n_jobs=1)
        
        mean_test_accuracy = np.mean(scores['test_accuracy'])
        test_time = np.mean(scores['score_time'])

        # Unakrsna validacija može vratiti NaN vrijednosti, pogledati GitHub issue #27180
        if np.isnan(mean_test_accuracy):
            mean_test_accuracy = 0.000001
        if np.isnan(test_time):
            test_time = 1000000

        if mean_test_accuracy <= 0:
            mean_test_accuracy = 0.0000001
        if test_time <= 0:
            test_time = 0.0000001
        
        results.append({
            'Algoritam': name,
            'PreciznostAlgoritma': mean_test_accuracy,
            'VrijemeIzvrsavanja': test_time
        })

    df_results = pd.DataFrame(results)
    print("Performanse algoritama za trenutni skup podataka: " + '\n' +  str(df_results))
    
    # Izracunaj ARR vrijednost za svaki algoritam
    arr_values = []
    for i, row_i in df_results.iterrows():
        arr_sum = 0
        for j, row_j in df_results.iterrows():
            if i != j:
                arr = (row_i['PreciznostAlgoritma'] / row_j['PreciznostAlgoritma']) / (1 + alpha * np.log(row_i['VrijemeIzvrsavanja'] / row_j['VrijemeIzvrsavanja']))
                arr_sum += arr
        arr_values.append(arr_sum / (len(df_results) - 1))

    result = dict(zip(df_results['Algoritam'], arr_values))
    print("Konacna ARR mjera algoritama za trenutni skup podataka sa alfa vrijednosti " + str(alpha) + " : " +  str(result))

    return result

# Metoda za izracunavanje meta-znacajki u skupe podataka
def compute_metafeatures(X, y):
    mfe = MFE(groups="all", features=["attr_ent", "joint_ent"], summary=["median", "min", "max"])
    mfe.fit(X, y)
    features, values = mfe.extract()
    
    metafeatures = dict(zip(features, values))
    return metafeatures

def preporuciAlgoritam(targetX, targetY, alpha=0.05):

    # Učitaj skupove podataka
    pumpkins_data=pd.read_excel('Datasets/PumpkinSeed.xlsx')

    url_breast_cancer = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    column_names_bc = ["ID", "Diagnosis"] + ["Feature_" + str(i) for i in range(1, 31)]
    breast_cancer_data = pd.read_csv(url_breast_cancer, header=None, names=column_names_bc)

    url_wine = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
    column_names_wine = ["Class"] + ["Feature_" + str(i) for i in range(1, 14)]
    wine_data = pd.read_csv(url_wine, header=None, names=column_names_wine)

    url_parkinsons = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
    parkinsons_data = pd.read_csv(url_parkinsons)

    url_liver = "https://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv"
    column_names_liver = ["Age", "Gender", "TB", "DB", "Alkphos", "Sgpt", "Sgot", "TP", "ALB", "A/G Ratio", "Selector"]
    liver_data = pd.read_csv(url_liver, header=None, names=column_names_liver)

    url_sonar = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
    column_names_sonar = ["Feature_" + str(i) for i in range(1, 61)] + ["Class"]
    sonar_data = pd.read_csv(url_sonar, header=None, names=column_names_sonar)

    # Definiraj imena ciljnih varijabli
    target_variable_names = {
        "pumpkins_data": "Class",
        "breast_cancer_data": "Diagnosis",
        "wine_data": "Class",
        "parkinsons_data": "status",
        "liver_data": "Selector",
        "sonar_data": "Class"
    }

    # Početno uredi skupove podataka
    pumpkins_data['Class']=pumpkins_data['Class'].replace({'Çerçevelik':0,'Ürgüp Sivrisi':1})
    breast_cancer_data['Diagnosis']=breast_cancer_data['Diagnosis'].replace({'B':0,'M':1})
    parkinsons_data.drop(['name'],axis=1,inplace=True)
    liver_data['Gender']=liver_data['Gender'].replace({'Male':0,'Female':1})
    liver_data.dropna(subset=['A/G Ratio'], inplace=True)
    sonar_data['Class']=sonar_data['Class'].replace({'R':0,'M':1})

    datasets_list = [pumpkins_data, breast_cancer_data, wine_data, parkinsons_data, liver_data, sonar_data]
    for dataset in datasets_list:
        dataset.drop_duplicates(inplace=True)

    # Definiraj skup algoritama koji su kandidati za rjesenje problema selekcije algoritama
    algoritmi = {
        'LR': LogisticRegression(solver='lbfgs', max_iter=10000),
        'LDA': LinearDiscriminantAnalysis(),
        'CART': DecisionTreeClassifier(),
        "NB": GaussianNB(),
        "SVM": SVC(probability=True)
    }

    #Obradi skupove podataka i pripremi ih za treniranje
    preprocessed_datasets = []
    i = 0
    for df in datasets_list:
        y = df[list(target_variable_names.values())[i]].values
        X = df.drop(columns=[list(target_variable_names.values())[i]])
        preprocessed_datasets.append((X, y))
        i = i +1

    arr_values_list = []

    # Izracunaj ARR za svaki dataset
    for X, y in preprocessed_datasets:
        arr_values = compute_arr_values_for_dataset(X, y, algoritmi, alpha)
        arr_values_list.append(arr_values)

    best_algorithms = []

    # Pronadi najbolji algoritam za svaki dataset
    for arr_values in arr_values_list:
        best_algorithm = max(arr_values, key=arr_values.get)
        best_value = arr_values[best_algorithm]
        best_algorithms.append({best_algorithm: best_value})

    metafeatures_with_best_algo = []

    # Izracunaj meta-znacajke za svaki skup podataka i dodaj najbolji algoritam
    for (X, y), best_algo in zip(preprocessed_datasets, best_algorithms):
        metafeatures = compute_metafeatures(X.to_numpy(), y)
        metafeatures["bestAlg"] = list(best_algo.keys())[0]
        metafeatures_with_best_algo.append(metafeatures)

    # Pretvori imena algoritama u numericke vrijednosti
    algorithm_names = [meta["bestAlg"] for meta in metafeatures_with_best_algo]
    label_encoder = LabelEncoder()
    encoded_algo_names = label_encoder.fit_transform(algorithm_names)
    meta_df = pd.DataFrame(metafeatures_with_best_algo)

    X_meta = meta_df.drop("bestAlg", axis=1)  
    y_meta = encoded_algo_names
    X_meta_np = X_meta.to_numpy()

    # Treniraj Random Forest klasifikator na meta-znacajkama
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_meta_np, y_meta)

    metafeatures = compute_metafeatures(targetX, targetY)

    valuesNp = np.array(list(metafeatures.values())).reshape(1, -1)
    predicted_indices = rf.predict(valuesNp)
    predicted_algorithm_names = label_encoder.inverse_transform(predicted_indices)
    
    return predicted_algorithm_names