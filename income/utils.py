from collections import Counter
# index 찾기
mis_var_features = ['workclass', 'occupation', 'native.country']
def findMissIndex(df, mis_var_feature):
    index = data[data[mis_var_feature].isnull()].index
    return index

# mis_work_indices = findMissIndex(data, mis_var_features[0])
# mis_occ_indices = findMissIndex(data, mis_var_features[1])
# mis_con_indices = findMissIndex(data, mis_var_features[2])