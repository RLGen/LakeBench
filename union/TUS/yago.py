import re
import sqlite3


def process_string(input_string):
    # 将字符串中的字母转换为小写
    input_string = str(input_string)
    processed_string = input_string.lower()
    # 使用正则表达式将非字母数字字符替换为空格
    processed_string = re.sub(r'[^a-zA-Z0-9]', '', processed_string)
    return processed_string


def create_taxonomy_from_tsv(file_path, cursor):
    #创建表
    cursor.execute('''
        CREATE TABLE yagoTaxonomy (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subclass TEXT,
            relation TEXT,
            taxonomy_class TEXT
        )
    ''')
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            #处理读进来的line，如果第一列为空，就添加<t>
            if line.startswith('\t'):
            # 在字符串头部添加"<t>"
                line = "<t>" + line
            _, subclass, relation, taxonomy_class = line.strip().split('\t')
            cursor.execute('INSERT INTO yagoTaxonomy (subclass, relation, taxonomy_class) VALUES (?, ?, ?)', (subclass[1:-1], relation, taxonomy_class[1:-1]))


def create_type_from_tsv(file_path, cursor):
    cursor.execute('''
        CREATE TABLE yagoTypes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity TEXT,
            relation TEXT,
            type TEXT
        )
    ''')
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            _, entity, relation, type = line.strip().split('\t')
            #还得去掉头尾的<, 并且要把字母全部小写，把标点符号替换成空格,只替换entity的就行
            entity = process_string(entity[1:-1])
            cursor.execute('INSERT INTO yagoTypes (entity, relation, type) VALUES (?, ?, ?)', (entity, relation, type[1:-1]))


def create_type_join_class_from_table(cursor):
    cursor.execute('''
        Create Table yagoTypeJoinClasses(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity TEXT,
            type TEXT,
            taxonomy_class TEXT
        )
        '''
    )

    cursor.execute('''
        INSERT INTO yagoTypeJoinClasses (entity, type, taxonomy_class)
        SELECT y.entity, y.type, t.taxonomy_class
        FROM yagotypes y
        JOIN yagoTaxonomy t ON y.type = t.subclass;
    ''')

def create_entitys_from_yagotypes(cursor):
    cursor.execute('''
        Create virtual Table if not exists yagoEntitys using fts5(
            entity
        )
    ''')

    cursor.execute('''
        INSERT INTO yagoEntitys (entity)
        SELECT distinct entity
        FROM yagotypes 
    ''')


def main():
    type_tsv_file = 'yago/yagoTypes.tsv'  #tsv文件路径，这里面是实体和类的对应关系
    taxonomy_tsv_file = 'yago/yagoTaxonomy.tsv'
    
    conn = sqlite3.connect('yago/yago.db')
    cursor = conn.cursor()
    
    cursor.close()
    conn.close()

    create_type_from_tsv(type_tsv_file, cursor)
    conn.commit()
    print('type ok')

    create_taxonomy_from_tsv(taxonomy_tsv_file, cursor)
    conn.commit()
    print('taxonomy ok')

    create_type_join_class_from_table(cursor)
    conn.commit()
    print('all ok!')

    # create_entitys_from_yagotypes(cursor)
    # conn.commit()
    # print('entitys ok')
    # search_keywords = "a b c d"
    # query = f"SELECT distinct entity FROM yagoTypes WHERE entity MATCH '{search_keywords}' order by bm25(yagotypes) desc limit 10"
    # cursor.execute('SELECT * FROM yagoTypes LIMIT 10')
    # cursor.execute(query)
    # results = cursor.fetchall()

    # for row in results:
    #     print(row)

    cursor.execute('SELECT * FROM yagoEntitys LIMIT 10')
    results = cursor.fetchall()
    for row in results:
        print(row)

    cursor.execute('SELECT * FROM yagoTaxonomy LIMIT 10')
    results = cursor.fetchall()
    for row in results:
        print(row)
    
    conn.commit()
    # 关闭游标和数据库连接
    cursor.close()
    conn.close()


def get_yagoTypes_from_tsv():
    #创建虚拟表，在这个虚拟表上可以执行match以及bm25
    type_tsv_file = 'yago/yagoTypes.tsv' 
    conn = sqlite3.connect('yago/yago.db')
    cursor = conn.cursor()
    #create_type_from_tsv(type_tsv_file, cursor)
    #conn.commit()

    return conn, cursor

#把join表重新建立，然后建立多线程
if __name__ == "__main__":
    main()