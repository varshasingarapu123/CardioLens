AAMI_GROUPS={'N':'N','L':'N','R':'N','e':'N','j':'N','A':'S','a':'S','J':'S','S':'S','V':'V','E':'V','F':'F','/':'Q','f':'Q','Q':'Q','?':'Q'}
CLASS_NAMES={0:'N',1:'S',2:'V',3:'F',4:'Q'}
CLASS_TO_ID={v:k for k,v in CLASS_NAMES.items()}
def map_symbol(symbol): return AAMI_GROUPS.get(symbol)
def class_id(symbol):
    mapped=map_symbol(symbol)
    return None if mapped is None else CLASS_TO_ID[mapped]
