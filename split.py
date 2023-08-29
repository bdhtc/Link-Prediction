if __name__ == '__main__':
    import numpy as np
    from sklearn.model_selection import train_test_split

    # 以'utf-8'编码方式读取文件
    data = np.genfromtxt('CoOccur.txt', delimiter='\t', dtype=str, encoding='utf-8')

    # 提取前三列数据作为特征
    features = data[:, :-1]

    # 提取第四列数据作为年份
    years = data[:, -1]

    # 将数据划分为训练集和测试集，使用10比1的比例
    train_features, test_features, train_years, test_years = train_test_split(features, years, test_size=0.1,
                                                                              random_state=42)

    # 将训练集和测试集的特征和年份合并为一个新的数组
    train_data = np.column_stack((train_features, train_years))
    test_data = np.column_stack((test_features, test_years))

    # 保存训练集和测试集为两个不同的txt文件
    np.savetxt('CoOccur_train_data.txt', train_data, fmt='%s', delimiter='\t')
    np.savetxt('CoOccur_test_data.txt', test_data, fmt='%s', delimiter='\t')

