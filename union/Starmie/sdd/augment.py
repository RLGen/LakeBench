import pandas as pd
import random

def augment(table: pd.DataFrame, op: str):
    """Apply an augmentation operator on a table.

    Args:
        table (DataFrame): the input table
        op (str): operator name
    
    Return:
        DataFrame: the augmented table
    """
    if op == 'drop_col':
        # set values of a random column to 0
        col = random.choice(table.columns)
        table = table.copy()
        table[col] = ""
    elif op == 'sample_row':
        # sample 50% of rows
        if len(table) > 0:
            table = table.sample(frac=0.5)
    elif op == 'sample_row_ordered':
        # sample 50% of rows
        if len(table) > 0:
            table = table.sample(frac=0.5).sort_index()
    elif op == 'shuffle_col':
        # shuffle the column orders
        new_columns = list(table.columns)
        random.shuffle(new_columns)
        table = table[new_columns]
    elif op == 'drop_cell':
        # drop a random cell
        table = table.copy()
        row_idx = random.randint(0, len(table) - 1)
        col_idx = random.randint(0, len(table.columns) - 1)
        table.iloc[row_idx, col_idx] = ""
    elif op == 'sample_cells':
        # sample half of the cells randomly
        table = table.copy()
        col_idx = random.randint(0, len(table.columns) - 1)
        sampleRowIdx = []
        for _ in range(len(table) // 2 - 1):
            sampleRowIdx.append(random.randint(0, len(table) - 1))
        for ind in sampleRowIdx:
            table.iloc[ind, col_idx] = ""
    elif op == 'replace_cells':
        # replace half of the cells randomly with the first values after sorting
        table = table.copy()
        col_idx = random.randint(0, len(table.columns) - 1)
        sortedCol = table[table.columns[col_idx]].sort_values().tolist()
        sampleRowIdx = []
        for _ in range(len(table) // 2 - 1):
            sampleRowIdx.append(random.randint(0, len(table) - 1))
        for ind in sampleRowIdx:
            table.iloc[ind, col_idx] = sortedCol[ind]
    elif op == 'drop_head_cells':
        # drop the first quarter of cells
        table = table.copy()
        col_idx = random.randint(0, len(table.columns) - 1)
        sortedCol = table[table.columns[col_idx]].sort_values().tolist()
        sortedHead = sortedCol[:len(table)//4]
        for ind in range(0,len(table)):
            if table.iloc[ind, col_idx] in sortedHead:
                table.iloc[ind, col_idx] = ""
    elif op == 'drop_num_cells':
        # drop numeric cells
        table = table.copy()
        tableCols = list(table.columns)
        numTable = table.select_dtypes(include=['number'])
        numCols = numTable.columns.tolist()
        if numCols == []:
            col_idx = random.randint(0, len(table.columns) - 1)
        else:
            col = random.choice(numCols)
            col_idx = tableCols.index(col)
        sampleRowIdx = []
        for _ in range(len(table) // 2 - 1):
            sampleRowIdx.append(random.randint(0, len(table) - 1))
        for ind in sampleRowIdx:
            table.iloc[ind, col_idx] = ""
    elif op == 'swap_cells':
        # randomly swap two cells
        table = table.copy()
        row_idx = random.randint(0, len(table) - 1)
        row2_idx = random.randint(0, len(table) - 1)
        while row2_idx == row_idx:
            row2_idx = random.randint(0, len(table) - 1)
        col_idx = random.randint(0, len(table.columns) - 1)
        cell1 = table.iloc[row_idx, col_idx]
        cell2 = table.iloc[row2_idx, col_idx]
        table.iloc[row_idx, col_idx] = cell2
        table.iloc[row2_idx, col_idx] = cell1
    elif op == 'drop_num_col': # number of columns is not preserved
        # remove numeric columns
        numTable = table.select_dtypes(include=['number'])
        numCols = numTable.columns.tolist()
        textTable = table.select_dtypes(exclude=['number'])
        textCols = textTable.columns.tolist()
        addedCols = 0
        while addedCols <= len(numCols) // 2 and len(numCols) > 0:
            numRandCol = numCols.pop(random.randrange(len(numCols)))
            textCols.append(numRandCol)
            addedCols += 1
        textCols = sorted(textCols,key=list(table.columns).index)
        table = table[textCols]
    elif op == 'drop_nan_col': # number of columns is not preserved
        # remove a half of the number of columns that contain nan values
        newCols, nanSums = [], {}
        for column in table.columns:
            if table[column].isna().sum() != 0:
                nanSums[column] = table[column].isna().sum()
            else:
                newCols.append(column)
        nanSums = {k: v for k, v in sorted(nanSums.items(), key=lambda item: item[1], reverse=True)}
        nanCols = list(nanSums.keys())
        newCols += random.sample(nanCols, len(nanCols) // 2)
        table = table[newCols]
    elif op == 'shuffle_row': 
        # shuffle the rows
        table = table.sample(frac=1)

    return table