def grid_binary(model, params, data, target, cv=2):
    
    # split
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=442)
    
    # train
    
    print("GridSeachCV proceeding...")
    gbm = GridSearchCV(model, params, n_jobs=-1, cv=cv, verbose=1)
    gbm.fit(X_train, y_train)
    print("Done.")
    
    print(101*"="+ "\nBEST PARAMETERS: ", gbm.best_params_, "\n"+101*"="+"\n")
    predictions = gbm.predict(X_test)
    
    print("==================\AUC : %.4g\n==================" % metrics.roc_auc_score(y_test, predictions))
