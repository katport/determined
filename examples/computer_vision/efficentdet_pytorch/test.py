class DotDict(dict):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    # __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


    def __init__(self, dct):
        for key, value in dct.items():
            # if hasattr(value, 'keys'):
            #     value = DotDict(value)
            self[key] = value
    def __getattr__(self, name):
        try:
            return self[name]
        except: 
            return None
# from dotmap import DotMap
# d = DotMap(t)

t = {'a': 1, 'b': 2}

d = DotDict(t)

print ('be 1: ', d.a)
print ('be false: ', hasattr(d, 'p'))

print (getattr(d, 'lr_noise', None))
print (getattr(d, 'lr_noise', None) is not None)
