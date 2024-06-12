import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

pd.set_option('future.no_silent_downcasting', True)

def data_adjustment():

    # Reading data --------------------------------------------------------------------------------------------------------------------------------
    dfPenguins = pd.read_csv('palmerpenguins.csv', delimiter=',')

    # Mapping for data replacement ----------------------------------------------------------------------------------------------------------------
    island_dict = {"Biscoe": 0, "Dream": 1, "Torgersen": 2}
    sex_dict = {"FEMALE": 0, "MALE": 1}
    species_dict = {"Adelie": 0, "Chinstrap": 1, "Gentoo": 2}

    # Replacing data with its respective values
    dfPenguins['island'] = dfPenguins['island'].replace(island_dict).astype(int)
    dfPenguins['species'] = dfPenguins['species'].replace(species_dict).astype(int)
    dfPenguins['sex'] = dfPenguins['sex'].replace(sex_dict).astype(int)

    # Reordering collums --------------------------------------------------------------------------------------------------------------------------
    dfPenguins = dfPenguins.reindex(columns=['island', 'sex', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'species'])

    return dfPenguins


def data_training(dfPenguins: pd.DataFrame):
    
    # Splitting data ------------------------------------------------------------------------------------------------------------------------------
    X = dfPenguins.drop('species', axis=1)
    y = dfPenguins['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # KNN Training --------------------------------------------------------------------------------------------------------------------------------
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)

    # KNN Prediction
    y_predict_KNN = knn.predict(X_test)

    # Printing KNN accuracy results
    reportKNN = classification_report(y_test, y_predict_KNN)
    print(reportKNN)

    # Decision Tree Training ----------------------------------------------------------------------------------------------------------------------
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)

    # Decision Tree Prediction
    y_predict_DT = dt.predict(X_test)

    # Printing Decision Tree accuracy results
    reportDT = classification_report(y_test, y_predict_DT)    
    print(reportDT)

    return knn, dt

if __name__ == "__main__":

    # Data processing and training
    dfPenguins = data_adjustment()
    knn, dt = data_training(dfPenguins)

    while(1):
        # User input -------------------------------------------------------------------------------------------------------------------------------
        print("\nEnter the penguins data: ")
        penguins_number = int(input("\nNumber of penguins: "))

        penguin_list = []

        for i in range(penguins_number):
            print("\nPenguin ", i+1)
            island = int(input("\nIsland (0 - Biscoe, 1 - Dream, 2 - Torgersen): "))
            sex = int(input("Sex (0 - Female, 1 - Male): "))
            culmen_length = float(input("Culmen Length (mm): "))
            culmen_depth = float(input("Culmen Depth (mm): "))
            flipper_length = float(input("Flipper Length (mm): "))
            body_mass = float(input("Body Mass (g): "))

            penguin_list.append([island, sex, culmen_length, culmen_depth, flipper_length, body_mass])

        dfAnalysis = pd.DataFrame(penguin_list, columns=['island', 'sex', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g'])

        # Predicting penguins species ----------------------------------------------------------------------------------------------------------------
        penguin_predict_KNN = knn.predict(dfAnalysis)
        penguin_predict_DT = dt.predict(dfAnalysis)

        species_dict = {0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}

        print("\nResults: ")
        for i in range(penguins_number):
            print("\nPenguin ", i+1)
            print("KNN Prediction: ", species_dict[penguin_predict_KNN[i]])
            print("Decision Tree Prediction: ", species_dict[penguin_predict_DT[i]])



