from config import *

#train = pd.read_csv('../input/train.csv')
train = pd.read_csv('../')
test = pd.read_csv('../input/test.csv')
##追加のデータ
##より深く学習させる
#external_mlqa = pd.read_csv('../input/external_data/mlqa_hindi.csv')
#external_xquad = pd.read_csv('../input/external_data/xquad.csv')
#tamil_xquad = pd.read_csv('../input//squad_translated_tamil.csv')
# external_train = pd.concat([external_mlqa, external_xquad, tamil_xquad, tamil_squad4])


def create_folds(data, num_splits):
    data["kfold"] = -1
    kf = model_selection.StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=2021)
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data['language'])):
        data.loc[v_, 'kfold'] = f
    return data


if __name__ == "__main__":
    external_train = pd.concat([external_mlqa, external_xquad, tamil_xquad])
    train = create_folds(train, num_splits=5)
    external_train["kfold"] = -1
    external_train['id'] = list(np.arange(1, len(external_train)+1))
    train = pd.concat([train, external_train]).reset_index(drop=True)
    train.to_csv('folded_inputs.csv', index=False)



