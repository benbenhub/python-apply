def get_sample(df, sampling = "sample_random", k = 1, stratified_col = None):
    """
    对输入的dataframe进行抽样的函数
    参数：
        - df:输入的数据框pandas.dataframe对象
        - samlping:抽样方法,str类型
            而可选值有["simple_random","stratified","systematic"]
            分别为简单随机抽样,分层抽样,系统抽样
        - k:抽样个数或抽样比例,int或float类型
            若为int类型,则k必须大于0;若为float类型,则k必须在(0, 1)中
            如果0<k<1,则k表示抽样占总体的比例
            如果k>=1,则k表示抽样的个数;当为分层抽样时,k表示每一层的样本量
        - stratified_col:需要分层的列名的列表,list类型只有在分层抽样时才生效

    返回值:
        pandas.dataframe对象,抽样结果
    """
    import random
    import pandas as pd
    from functools import reduce
    import numpy as np
    import math

    len_df = len(df)
    if k <= 0:
        raise AssertionError("k不能为负数")
    elif k >= 1:
        assert isinstance(k, int), "在选择抽样个数时,k必须为正整数"
        sample_by_n = True
        if sampling is "stratified":
            alln = k * df.groupby(by = stratified_col)[stratified_col[0]].count() 
            if alln >= len_df:
                raise AssertionError("请确认k乘以层数不能超过总样本量")
            else:
                sample_by_n = False
                if sampling in ("simple_random", "systematic"):
                    k = math.ceil(len_df * k)

            if sampling is "simple_random":
                print("使用简单随机抽样")
                idx = random.sample(range(len_df), k)
                res_df = df.iloc[idx, :].copy()
                return res_df
            
            elif sampling is "systematic":
                print("使用系统抽样")
                step = len_df // k+1
                start = 0
                idx = range(len_df)[start::step]
                res_df = df.iloc[idx, :].copy()
                return res_df
            
            elif sampling is "stratified":
                assert stratified_col is not None, "请传入包含需要分层的列名的列表"
                assert all(np.inld(stratified_col, df.columns)), "请检查输入的列名"
                grouped = df.groupby(by = stratified_col)[stratified_col[0]].count()
                if sample_by_n == True:
                    group_k = grouped.map(lambda x:k)
                else:
                    group_k = grouped.mpa(lambda x: math.ceil(x * k))

                res_df = pd.DataFrame(columns=df.columns)
                for df_idx in group_k.index:
                    df1=df
                    if len(stratified_col) == 1:
                        df1 = df1[df1[stratified_col[0]] == df_idx]
                    else:
                        for i in range(len(df_idx)):
                            df1 = df1[df1[stratified_col[i]] == df_idx[i]]
                    
                    idx = random.sample(range(len(df1)), group_k[df_idx])
                    group_df = df1.iloc[idx, :].copy()
                    res_df = res_df.append(group_df)
                return res_df
            else:
                raise AssertionError("sampling is illegal")