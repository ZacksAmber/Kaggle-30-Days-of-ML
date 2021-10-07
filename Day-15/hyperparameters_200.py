class Model_Blending:
    def __init__(self, train_path):
        import warnings
        warnings.filterwarnings('ignore')

        # Import datasets
        self.df_train = pd.read_csv(train_path)
        self.df_test = pd.read_csv('../input/30-days-of-ml/test.csv')
        #self.df_test = pd.read_csv('data/test.csv')
        self.sample_submission = pd.read_csv('../input/30-days-of-ml/sample_submission.csv')
        
        # Define features
        self.num_cols = ['cont0', 'cont1', 'cont2', 'cont3', 'cont4', 'cont5', 'cont6', 'cont7', 'cont8', 'cont9', 'cont10', 'cont11', 'cont12', 'cont13']
        self.onehot_cols = ['cat0', 'cat1', 'cat3', 'cat5', 'cat6', 'cat7', 'cat8'] # remove 'cat2', 'cat4' due to the low MI scores
        self.ordinal_cols = ['cat9']
        self.cat_cols = self.onehot_cols + self.ordinal_cols
        self.useful_features = self.num_cols + self.cat_cols
        self.target = 'target'
    
    # Preprocessing solution 0: Ordinal Encoding
    def _ordinal_encoding(self, X_train, X_valid, X_test, params=True):
        # Preprocessing - Ordinal Encoding
        oe = OrdinalEncoder()
        X_train[self.cat_cols] = oe.fit_transform(X_train[self.cat_cols])
        X_valid[self.cat_cols] = oe.transform(X_valid[self.cat_cols])
        X_test[self.cat_cols] = oe.transform(X_test[self.cat_cols])
        
        """no_outliers
        # 0.7172987346930846
        # XGBoost params
        xgb_params = {
            'alpha': 7.128681031027614,
            'lambda': 0.40760576474680843,
            'gamma': 0.08704298132127238,
            'reg_alpha': 25.377502919374336,
            'reg_lambda': 0.003401041649454036,
            'colsample_bytree': 0.1355660282707954,
            'subsample': 0.6999406375783235,
            'learning_rate': 0.02338550339980208,
            'n_estimators': 9263,
            'max_depth': 6,
            'random_state': 2021,
            'min_child_weight': 138
        }

        # 0.7174088504920006
        # LightGBM params
        lgb_params = {
            'random_state': 0, 
            'num_iterations': 9530, 
            'learning_rate': 0.018509357813869098, 
            'max_depth': 6, 
            'num_leaves': 98, 
            'min_data_in_leaf': 1772, 
            'lambda_l1': 0.0010866230909549698, 
            'lambda_l2': 1.6105154171511057e-05, 
            'feature_fraction': 0.09911317646202211, 
            'bagging_fraction': 0.8840672050147438, 
            'bagging_freq': 6, 
            'min_child_samples': 35
        }
        """

        """Full dataset"""
        # 0.7168956185375995
        # XGBoost params
        xgb_params = {
            'alpha': 0.41478790863509063, 
            'lambda': 4.533806139098733, 
            'gamma': 0.006523052455552593, 
            'reg_alpha': 16.714567692323264, 
            'reg_lambda': 6.321580437513598e-06, 
            'colsample_bytree': 0.11544585116842096, 
            'subsample': 0.8448523684136955, 
            'learning_rate': 0.061677285578690844, 
            'n_estimators': 8676, 
            'max_depth': 3, 
            'random_state': 0, 
            'min_child_weight': 268
        }

        # 0.716864914502225
        # LightGBM params
        lgb_params = {
            'random_state': 42, 
            'num_iterations': 9718, 
            'learning_rate': 0.014607325219450438, 
            'max_depth': 6, 
            'num_leaves': 11, 
            'min_data_in_leaf': 704, 
            'lambda_l1': 1.1906873176743155e-07, 
            'lambda_l2': 1.0479700691744536, 
            'feature_fraction': 0.13449900909384252, 
            'bagging_fraction': 0.6680657144501343, 
            'bagging_freq': 1, 
            'min_child_samples': 66
        }

        
        if params == True:
            return X_train, X_valid, X_test, xgb_params, lgb_params
        else:
            return X_train, X_valid, X_test
    
    # Preprocessing solution 1: One-hot Encoding + Ordinal Encoding
    def _onehot_encoding(self, X_train, X_valid, X_test):
        # Preprocessing - One-hot Encoding
        ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
        X_train_ohe = ohe.fit_transform(X_train[self.onehot_cols])
        X_valid_ohe = ohe.transform(X_valid[self.onehot_cols])
        X_test_ohe = ohe.transform(X_test[self.onehot_cols])

        X_train_ohe = pd.DataFrame(X_train_ohe, columns=[f"ohe_{i}" for i in range(X_train_ohe.shape[1])])
        X_valid_ohe = pd.DataFrame(X_valid_ohe, columns=[f"ohe_{i}" for i in range(X_valid_ohe.shape[1])])
        X_test_ohe = pd.DataFrame(X_test_ohe, columns=[f"ohe_{i}" for i in range(X_test_ohe.shape[1])])

        X_train = pd.concat([X_train.drop(columns=self.onehot_cols), X_train_ohe], axis=1)
        X_valid = pd.concat([X_valid.drop(columns=self.onehot_cols), X_valid_ohe], axis=1)
        X_test = pd.concat([X_test.drop(columns=self.onehot_cols), X_test_ohe], axis=1)
        
        # Preprocessing - Ordinal Encoding
        oe = OrdinalEncoder()
        X_train[self.ordinal_cols] = oe.fit_transform(X_train[self.ordinal_cols])
        X_valid[self.ordinal_cols] = oe.transform(X_valid[self.ordinal_cols])
        X_test[self.ordinal_cols] = oe.transform(X_test[self.ordinal_cols])
    
        """No outliers
        # 0.7174931253475558
        # XGBoost params
        xgb_params = {
            'alpha': 3.046687193123841,
            'lambda': 0.7302844649944737,
            'gamma': 0.10108768743909796,
            'reg_alpha': 14.711350393993625,
            'reg_lambda': 1.6855306764481926e-07,
            'colsample_bytree': 0.15006790036326567,
            'subsample': 0.9761751211889541,
            'learning_rate': 0.02730958701307226,
            'n_estimators': 7897,
            'max_depth': 4,
            'random_state': 0,
            'min_child_weight': 203
        }
        
        # 0.7172624587909345
        # LightGBM params
        lgb_params = {
            'random_state': 42, 
            'num_iterations': 6969, 
            'learning_rate': 0.014404708757048168, 
            'max_depth': 7, 
            'num_leaves': 21, 
            'min_data_in_leaf': 1121, 
            'lambda_l1': 4.1636932334315094e-07, 
            'lambda_l2': 1.0975422991510602e-08, 
            'feature_fraction': 0.08082581387850206, 
            'bagging_fraction': 0.6804475225598854, 
            'bagging_freq': 2, 
            'min_child_samples': 32
        }
        """
 
        """Full dataset"""
        # 0.7169803941400036
        # XGBoost params
        xgb_params = {
            'alpha': 0.006431298298825313, 
            'lambda': 1.9946701540008387, 
            'gamma': 0.004468866854966971, 
            'reg_alpha': 1.7419809828857386e-08, 
            'reg_lambda': 0.25665090431195203, 
            'colsample_bytree': 0.10016560933147275, 
            'subsample': 0.770411681261352, 
            'learning_rate': 0.01350994278419047, 
            'n_estimators': 8378, 
            'max_depth': 5, 
            'random_state': 0, 
            'min_child_weight': 237
        }

        # 0.7168422881316736
        # LightGBM params
        lgb_params = {
            'random_state': 0, 
            'num_iterations': 9121, 
            'learning_rate': 0.03762205881915334, 
            'max_depth': 5, 
            'num_leaves': 12, 
            'min_data_in_leaf': 1331, 
            'lambda_l1': 1.4246377549177525e-08, 
            'lambda_l2': 0.00031572480246719195, 
            'feature_fraction': 0.07760290667911449, 
            'bagging_fraction': 0.9766045388889536, 
            'bagging_freq': 4, 
            'min_child_samples': 50
        }
        
        return X_train, X_valid, X_test, xgb_params, lgb_params

    # Preprocessing solution 2: Ordinal Encoding + Standardization
    def _standardization(self, X_train, X_valid, X_test):
        # Preprocessing - Standardization
        scaler = StandardScaler()
        X_train[self.num_cols] = scaler.fit_transform(X_train[self.num_cols])
        X_valid[self.num_cols] = scaler.transform(X_valid[self.num_cols])
        X_test[self.num_cols] = scaler.transform(X_test[self.num_cols])
    
        """No outliers
        # 0.7172152365762312
        # XGBoost params
        xgb_params = {
            'alpha': 0.029925179326119784,
            'lambda': 0.12530061860157662,
            'gamma': 0.5415753114227984,
            'reg_alpha': 14.992919845445886,
            'reg_lambda': 0.42076728548917974,
            'colsample_bytree': 0.10022710624560974,
            'subsample': 0.5596856445758918,
            'learning_rate': 0.020866717779139694,
            'n_estimators': 6852,
            'max_depth': 7,
            'random_state': 2021,
            'min_child_weight': 62
        }
        
        # 0.7173410652198884
        # LightGBM params
        lgb_params = {
            'random_state': 0,
            'num_iterations': 6439,
            'learning_rate': 0.03625416364918611,
            'max_depth': 6,
            'num_leaves': 11,
            'min_data_in_leaf': 745,
            'lambda_l1': 4.1932281223524115e-06,
            'lambda_l2': 0.043343249414638636,
            'feature_fraction': 0.08623933710228435,
            'bagging_fraction': 0.7934935001504152,
            'bagging_freq': 3,
            'min_child_samples': 23
        }
        """

        """Full dataset"""
        # 0.7169754780128185
        # XGBoost params
        xgb_params = {
            'alpha': 0.001954751535110173, 
            'lambda': 0.4889428995375083, 
            'gamma': 0.2639993537540112, 
            'reg_alpha': 4.613139964829376, 
            'reg_lambda': 2.5644939695394116e-06, 
            'colsample_bytree': 0.10081771115519686, 
            'subsample': 0.7241988095847515, 
            'learning_rate': 0.02383200558701416, 
            'n_estimators': 6995, 
            'max_depth': 4, 
            'random_state': 42, 
            'min_child_weight': 198
        }

        # 0.7168529799564377
        # LightGBM params
        lgb_params = {
            'random_state': 2021, 
            'num_iterations': 7614, 
            'learning_rate': 0.013334143201183738, 
            'max_depth': 7, 
            'num_leaves': 18, 
            'min_data_in_leaf': 1270, 
            'lambda_l1': 6.801685953700497e-05, 
            'lambda_l2': 6.178728858466813e-08, 
            'feature_fraction': 0.13830499588135214, 
            'bagging_fraction': 0.8060814091341896, 
            'bagging_freq': 3, 
            'min_child_samples': 81
        }
        
        return X_train, X_valid, X_test, xgb_params, lgb_params

    # Preprocessing solution 3: Ordinal Encoding + Log transformation
    def _log_transformation(self, X_train, X_valid, X_test):
        # Preprocessing - Log transformation
        for col in self.num_cols:
            X_train[col] = np.log1p(X_train[col])
            X_valid[col] = np.log1p(X_valid[col])
            X_test[col] = np.log1p(X_test[col])

        """No outliers
        # 0.7172539872780895
        # XGBoost params
        xgb_params = {
            'alpha': 0.08862033338686888, 
            'lambda': 0.003553846716302233, 
            'gamma': 0.4097695581309838, 
            'reg_alpha': 17.808150656220917, 
            'reg_lambda': 1.6112661145526217, 
            'colsample_bytree': 0.11935885763757494, 
            'subsample': 0.7326515814471944, 
            'learning_rate': 0.04006687786137418, 
            'n_estimators': 5239, 
            'max_depth': 5, 
            'random_state': 2021, 
            'min_child_weight': 258
        }

        # 0.7174737448879298
        # LightGBM params
        lgb_params = {
            'random_state': 0,
            'num_iterations': 7945,
            'learning_rate': 0.05205269244224801,
            'max_depth': 6,
            'num_leaves': 9,
            'min_data_in_leaf': 1070,
            'lambda_l1': 1.0744924634974802e-07,
            'lambda_l2': 1.1250360028635182,
            'feature_fraction': 0.10421484055936374,
            'bagging_fraction': 0.916143112009066,
            'bagging_freq': 6,
            'min_child_samples': 20
        }
        """

        """Full dataset"""
        # 0.7170384818940932
        # XGBoost params
        xgb_params = {
            'alpha': 0.017253367743182032, 
            'lambda': 1.2312523239198236, 
            'gamma': 0.8870836430062957, 
            'reg_alpha': 9.3294011692425e-08, 
            'reg_lambda': 0.080494664534471, 
            'colsample_bytree': 0.12723293204878566, 
            'subsample': 0.5562373818875186, 
            'learning_rate': 0.01759177927013953, 
            'n_estimators': 7480, 
            'max_depth': 5, 
            'random_state': 42, 
            'min_child_weight': 216
        }

        # 0.716867820750228
        # LightGBM params
        lgb_params = {
            'random_state': 2021, 
            'num_iterations': 9135, 
            'learning_rate': 0.05204126206296579, 
            'max_depth': 3, 
            'num_leaves': 32, 
            'min_data_in_leaf': 1196, 
            'lambda_l1': 0.07110967369867664, 
            'lambda_l2': 9.981527842388462e-08, 
            'feature_fraction': 0.13002087379571273, 
            'bagging_fraction': 0.6510683790039721, 
            'bagging_freq': 2, 
            'min_child_samples': 19
        }
        
        return X_train, X_valid, X_test, xgb_params, lgb_params

    # Preprocessing solution 4: Target Encoding
    def _target_encoding(self, X_train, X_valid, X_test, y_train):
        # Preprocessing - Target Encoding
        te = MEstimateEncoder(cols=self.cat_cols, m=8) # m is from previous step
        X_train = te.fit_transform(X_train, y_train)
        X_valid = te.transform(X_valid)
        X_test = te.transform(X_test)
    
        """No outliers
        # 0.7172617296722674
        # XGBoost params
        xgb_params = {
            'alpha': 0.012609024116174448,
            'lambda': 0.7990281671135536,
            'gamma': 0.16689280834519887,
            'reg_alpha': 16.48576968441873,
            'reg_lambda': 4.83082534682402e-08,
            'colsample_bytree': 0.1162304168345657,
            'subsample': 0.9126362948665406,
            'learning_rate': 0.05528416190414117,
            'n_estimators': 9670,
            'max_depth': 5,
            'random_state': 42,
            'min_child_weight': 280
         }

        # 0.7173917173794985
        # LightGBM params
        lgb_params = {
            'random_state': 2021, 
            'num_iterations': 7977, 
            'learning_rate': 0.01618931564625682, 
            'max_depth': 5, 
            'num_leaves': 50, 
            'min_data_in_leaf': 890, 
            'lambda_l1': 0.003233614433753064, 
            'lambda_l2': 2.0001872037801434e-06, 
            'feature_fraction': 0.13638848986185334, 
            'bagging_fraction': 0.7045068716734475, 
            'bagging_freq': 2, 
            'min_child_samples': 79
        }
        """
        
        """Full dataset"""
        # 0.7169975934181431
        # XGBoost params
        xgb_params = {
            'alpha': 0.1211523885965823,
            'lambda': 0.00452864739396485,
            'gamma': 0.03948208038791913,
            'reg_alpha': 12.908845680463497,
            'reg_lambda': 3.120894405337636e-07,
            'colsample_bytree': 0.10449109016850185,
            'subsample': 0.7633088122674517,
            'learning_rate': 0.03246721588939738,
            'n_estimators': 6866,
            'max_depth': 5,
            'random_state': 0,
            'min_child_weight': 131
         }

        # 0.7169507121946433
        # LightGBM params
        lgb_params = {
            'random_state': 2021, 
            'num_iterations': 5027, 
            'learning_rate': 0.05104410422762626, 
            'max_depth': 3, 
            'num_leaves': 77, 
            'min_data_in_leaf': 440, 
            'lambda_l1': 0.05579764755559036, 
            'lambda_l2': 4.375963929072086e-08, 
            'feature_fraction': 0.14611731889635768, 
            'bagging_fraction': 0.9005654268392156, 
            'bagging_freq': 1, 
            'min_child_samples': 23
        }
        
        return X_train, X_valid, X_test, xgb_params, lgb_params
    
    def _xgboost_reg(self, xgb_params):
        model = XGBRegressor(
                    tree_method='gpu_hist',
                    gpu_id=0,
                    predictor='gpu_predictor',
                    n_jobs=-1,
                    **xgb_params
                )
        return model
    
    def _lightgbm_reg(self, lgb_params):
        model = LGBMRegressor(
                    device='gpu',
                    gpu_platform_id=0,
                    gpu_device_id=0,
                    n_jobs=-1,
                    metric='rmse',
                    **lgb_params
                )
        return model
    
    def blending(self, model: str):
        '''Model blending. Generate 5 predictions according to 5 data preprocessing solutions.
        
        Args:
            model: One of xgboost or lightgbm
            
        Returns:
            None
        '''
        assert model in ['xgboost', 'lightgbm'], "ValueError: model must be one of ['xgboost', 'lightgbm']!"
        
        # Loop preprocessing solutions
        for preprocessing_solution in range(5):
            final_valid_predictions = {} # store final predictions of X_valid for each preprocessing_solution
            final_test_predictions = [] # store final predictions of X_test for each preprocessing_solution
            scores = [] # store RMSE scores for each preprocessing_solution
            print(f"Data Preprocessing Solution: {preprocessing_solution}, Model: {model}")
            print(f"Training ...")
            # Loop KFolds
            for fold in range(5):
                # Data Preprocessing
                X_train = self.df_train[self.df_train.kfold != fold].reset_index(drop=True)
                X_valid = self.df_train[self.df_train.kfold == fold].reset_index(drop=True)
                X_test = self.df_test.copy()
                
                # get X_valid id
                X_valid_ids = X_valid.id.values.tolist()
                
                y_train = X_train.pop(self.target)
                X_train = X_train[self.useful_features] # not include id, cat2, cat4
                y_valid = X_valid.pop(self.target)
                X_valid = X_valid[self.useful_features] # not include id, cat2, cat4
                X_test = X_test[self.useful_features]
                
                # Ordinal Encoding
                if preprocessing_solution == 0:
                    X_train, X_valid, X_test, xgb_params, lgb_params = self._ordinal_encoding(X_train, X_valid, X_test)
                # One-hot Encoding + Ordinal Encoding
                elif preprocessing_solution == 1:
                    X_train, X_valid, X_test, xgb_params, lgb_params = self._onehot_encoding(X_train, X_valid, X_test)
                # Ordinal Encoding + Standardization
                elif preprocessing_solution == 2:
                    X_train, X_valid, X_test = self._ordinal_encoding(X_train, X_valid, X_test, params=False)
                    X_train, X_valid, X_test, xgb_params, lgb_params = self._standardization(X_train, X_valid, X_test)
                # Ordinal Encoding + Log Transformation
                elif preprocessing_solution == 3:
                    X_train, X_valid, X_test = self._ordinal_encoding(X_train, X_valid, X_test, params=False)
                    X_train, X_valid, X_test, xgb_params, lgb_params = self._log_transformation(X_train, X_valid, X_test)
                # Target Encoding
                elif preprocessing_solution == 4:
                    X_train, X_valid, X_test, xgb_params, lgb_params = self._target_encoding(X_train, X_valid, X_test, y_train)
                
                # Define model
                if model == 'xgboost':
                    reg = self._xgboost_reg(xgb_params)
                elif model == 'lightgbm':
                    reg = self._lightgbm_reg(lgb_params)
                
                # Modeling - Training
                reg.fit(
                    X_train, y_train, 
                    early_stopping_rounds=300,
                    eval_set=[(X_valid, y_valid)],
                    verbose=False
                )
                
                # Modeling - Evaluation and Inference
                valid_preds = reg.predict(X_valid)
                test_preds = reg.predict(X_test)
                
                final_valid_predictions.update(dict(zip(X_valid_ids, valid_preds))) # loop 5 times with different valid id
                final_test_predictions.append(test_preds) # loop 5 times and get the mean predictions for each row later

                rmse = mean_squared_error(y_valid, valid_preds, squared=False)
                scores.append(rmse)
                print(f'Data Preprocessing Solution: {preprocessing_solution}, Fold: {fold}, RMSE: {rmse}')
                
            # Export results
            final_valid_predictions = pd.DataFrame.from_dict(final_valid_predictions, orient="index").reset_index()
            final_valid_predictions.columns = ["id", f"{model}_{preprocessing_solution}_pred"]
            final_valid_predictions.to_csv(f"{model}_{preprocessing_solution}_valid_pred.csv", index=False)

            test_mean_preds = np.mean(np.column_stack(final_test_predictions), axis=1) # get the meam predictions for each row
            test_mean_preds = pd.DataFrame({'id': self.sample_submission.id, f"{model}_{preprocessing_solution}_pred": test_mean_preds})
            test_mean_preds.to_csv(f"{model}_{preprocessing_solution}_test_pred.csv", index=False)
            print(f'Average RMSE: {np.mean(scores)}, STD of RMSE: {np.std(scores)}')
            print('-----------------------------------------------------------------')
            
    def predict(self, data_path: str):
        '''Predict and generate submission.csv
        
        Args:
            data_path: The path of directory that stores all of the datasets after model blending.
            
        Returns:
            None
        '''
        # Define featurs
        xgb_features = [f"xgboost_{i}_pred" for i in range(5)]
        lgb_features = [f"lightgbm_{i}_pred" for i in range(5)]
        useful_features = xgb_features + lgb_features
        
        return useful_features
        
        # Import datasets
        df_train = pd.read_csv('../input/30days-folds/train_folds.csv')
        df_test = pd.read_csv('../input/30-days-of-ml/test.csv')
        #data_path = '../input/kaggle30daysofml/kaggle-30days-of-ml/full_dataset'
        
        # Join datasets on id
        for i in range(5):
            _df_valid = pd.read_csv(f'{data_path}/xgboost_{i}_valid_pred.csv')
            df_train = df_train.merge(_df_valid, on='id', how='left')
            _df_test = pd.read_csv(f'{data_path}/xgboost_{i}_test_pred.csv')
            df_test = df_test.merge(_df_test, on='id', how='left')

        for i in range(5):
            _df_valid = pd.read_csv(f'{data_path}/lightgbm_{i}_valid_pred.csv')
            df_train = df_train.merge(_df_valid, on='id', how='left')
            _df_test = pd.read_csv(f'{data_path}/lightgbm_{i}_test_pred.csv')
            df_test = df_test.merge(_df_test, on='id', how='left')

        return X_train
        
        # Modeling
        final_predictions = []
        scores = []
        for fold in range(5):
            X_train =  df_train[df_train.kfold != fold].reset_index(drop=True)
            X_valid = df_train[df_train.kfold == fold].reset_index(drop=True)
            X_test = df_test.copy()

            y_train = X_train.target
            y_valid = X_valid.target

            X_train = X_train[useful_features]
            X_valid = X_valid[useful_features]
            X_test = X_test[useful_features]

            # Modeling - Training
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Modeling - Evaluation and Inference
            valid_preds = model.predict(X_valid)
            test_preds = model.predict(X_test)
            final_predictions.append(test_preds)
            rmse = mean_squared_error(y_valid, valid_preds, squared=False)
            print(f'Fold: {fold}, RMSE: {rmse}')
            scores.append(rmse)
            
            # Generate submission.csv
            preds = np.mean(np.column_stack(final_predictions), axis=1)
            preds = pd.DataFrame({'id': self.sample_submission.id, 'target': preds})
            preds.to_csv('submission.csv', index=False)
        
        print('Generate submission.csv succeed!')
        print(f'Average RMSE: {np.mean(scores)}, STD of RMSE: {np.std(scores)}')