import psutil
import os
from createIndex import createTokenID 
from createRawTokens import createRawTokens

def api(cpath: str,
        save_root: str):
    
    createRawTokens(cpath, save_root)
    process = psutil.Process(os.getpid())
    print("当前内存占用：%s" % (process.memory_info().rss / 1024 / 1024/1024))
    
    createTokenID(save_root)
    
    return True