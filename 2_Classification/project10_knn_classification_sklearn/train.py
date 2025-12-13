from data import load_data
from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train, y_test = load_data()

model = KNeighborsClassifier(
    n_neighbors=5,      # K = 5
    metric="euclidean"  # distance formula
)

model.fit(X_train, y_train)

print("KNN model trained using sklearn")
