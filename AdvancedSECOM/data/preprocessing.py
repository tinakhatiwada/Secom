"""
AdvancedSECOM/data/preprocessing.py
DataLoaderMixin — load, split, and preprocess the SECOM dataset.
"""
from __future__ import annotations

from ..dependencies import (
    traceback, np, pd,
    KNNImputer, SimpleImputer,
    RobustScaler, StandardScaler,
    PCA, RFE, SelectFromModel, SelectKBest, f_classif,
    RandomForestClassifier, LocalOutlierFactor,
    train_test_split,
)


class DataLoaderMixin:
    """Mixin: data loading, train/cal/test split, and preprocessing."""

    def load_data(self) -> bool:
        try:
            print("Loading SECOM dataset …")
            data = pd.read_csv(self.data_dir / "secom.data",
                               sep=r"\s+", header=None)
            labels = pd.read_csv(self.data_dir / "secom_labels.data",
                                 sep=r"\s+", header=None)
            data["label"] = labels.iloc[:, 0]
            self.X = data.drop("label", axis=1)
            self.y = data["label"]
            print(f"Dataset shape : {data.shape}")
            print(f"Label dist    :\n{self.y.value_counts()}")
            if len(self.X) < 2000:
                print("⚠  Small dataset (<2 000). Use repeated k-fold.")
            self.y = self.y.map({-1: 0, 1: 1})
            cnt = self.y.value_counts()
            self.class_weight_ratio = cnt[0] / max(cnt.get(1, 1), 1)
            print(f"Imbalance : {self.class_weight_ratio:.2f}:1 | "
                  f"Minority: {cnt.get(1,0)/len(self.y)*100:.2f}%")
        except Exception as e:
            print(f"Load error: {e}"); traceback.print_exc(); return False
        return True

    def split_data(self, random_state=None):
        rs = self.random_state if random_state is None else random_state
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=rs, stratify=self.y)

        # Carve calibration slice from the training pool at real distribution,
        # BEFORE any resampling happens — keeps X_test completely untouched.
        self.X_train, self.X_cal, self.y_train, self.y_cal = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=rs, stratify=y_temp)

        print(f"\nTrain: {len(self.X_train)} samples "
              f"({self.y_train.sum()} failures, {self.y_train.mean():.2%})")
        print(f"Cal  : {len(self.X_cal)} samples "
              f"({self.y_cal.sum()} failures, {self.y_cal.mean():.2%})")
        print(f"Test : {len(self.X_test)} samples "
              f"({self.y_test.sum()} failures, {self.y_test.mean():.2%})")

    def advanced_preprocessing(self, use_knn_imputer=False, use_robust_scaler=True,
                                pca_components=None, feature_selection_method="rfe",
                                remove_correlated=True, correlation_threshold=0.9,
                                add_lof_score=False):
        print("\nPreprocessing (leakage-safe) …")

        # Impute
        imp = KNNImputer(n_neighbors=5) if use_knn_imputer else SimpleImputer(strategy="median")
        self.X_train = pd.DataFrame(imp.fit_transform(self.X_train))
        self.X_cal   = pd.DataFrame(imp.transform(self.X_cal))
        self.X_test  = pd.DataFrame(imp.transform(self.X_test))

        # Constant columns
        const = [c for c in self.X_train.columns
                 if self.X_train[c].nunique() <= 1 or self.X_train[c].std() < 1e-6]
        if const:
            self.X_train.drop(columns=const, inplace=True)
            self.X_cal.drop(columns=const, inplace=True)
            self.X_test.drop(columns=const, inplace=True)
            print(f"  Removed {len(const)} constant columns")

        # Correlated columns
        if remove_correlated:
            corr = self.X_train.corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            hc = [c for c in upper.columns if any(upper[c] > correlation_threshold)]
            if hc:
                self.X_train.drop(columns=hc, inplace=True)
                self.X_cal.drop(columns=hc, inplace=True)
                self.X_test.drop(columns=hc, inplace=True)
                print(f"  Removed {len(hc)} correlated columns")

        # Scale
        scaler = RobustScaler() if use_robust_scaler else StandardScaler()
        self.X_train = pd.DataFrame(scaler.fit_transform(self.X_train))
        self.X_cal   = pd.DataFrame(scaler.transform(self.X_cal))
        self.X_test  = pd.DataFrame(scaler.transform(self.X_test))

        # Feature selection
        n_sel = min(200, self.X_train.shape[1])
        if feature_selection_method == "rfe":
            print("  RFE feature selection …")
            sel = RFE(RandomForestClassifier(n_estimators=50, random_state=self.random_state),
                      n_features_to_select=n_sel)
            self.X_train = pd.DataFrame(sel.fit_transform(self.X_train, self.y_train))
            self.X_cal   = pd.DataFrame(sel.transform(self.X_cal))
            self.X_test  = pd.DataFrame(sel.transform(self.X_test))
        elif feature_selection_method == "kbest":
            sel = SelectKBest(f_classif, k=n_sel)
            self.X_train = pd.DataFrame(sel.fit_transform(self.X_train, self.y_train))
            self.X_cal   = pd.DataFrame(sel.transform(self.X_cal))
            self.X_test  = pd.DataFrame(sel.transform(self.X_test))
        elif feature_selection_method == "model_based":
            sel = SelectFromModel(
                RandomForestClassifier(n_estimators=100, random_state=self.random_state),
                threshold="median")
            self.X_train = pd.DataFrame(sel.fit_transform(self.X_train, self.y_train))
            self.X_cal   = pd.DataFrame(sel.transform(self.X_cal))
            self.X_test  = pd.DataFrame(sel.transform(self.X_test))

        if pca_components:
            pca = PCA(n_components=pca_components)
            self.X_train = pd.DataFrame(pca.fit_transform(self.X_train))
            self.X_cal   = pd.DataFrame(pca.transform(self.X_cal))
            self.X_test  = pd.DataFrame(pca.transform(self.X_test))
            print(f"  PCA variance: {pca.explained_variance_ratio_.sum():.3f}")

        if add_lof_score:
            lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
            lof.fit(self.X_train)
            self.X_train["lof_score"] = -lof.score_samples(self.X_train)
            self.X_cal["lof_score"]   = -lof.score_samples(self.X_cal)
            self.X_test["lof_score"]  = -lof.score_samples(self.X_test)
            self.X_train.columns = range(self.X_train.shape[1])
            self.X_cal.columns   = range(self.X_cal.shape[1])
            self.X_test.columns  = range(self.X_test.shape[1])

        self.feature_names = [f"Feature_{i}" for i in range(self.X_train.shape[1])]
        print(f"  Final shape — train: {self.X_train.shape}  "
              f"cal: {self.X_cal.shape}  test: {self.X_test.shape}")
        print(f"  Missing — train: {self.X_train.isnull().sum().sum()}  "
              f"cal: {self.X_cal.isnull().sum().sum()}  "
              f"test: {self.X_test.isnull().sum().sum()}")
