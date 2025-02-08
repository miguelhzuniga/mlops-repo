if __name__=="__main__":
    from joblib import load
    model = load("model.pkl")
    print(model.n_neighbors)