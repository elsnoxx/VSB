import time
import random
import hashlib

abeceda = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def vzorka():
    vysledok = list()
    for _ in range(0, 1000000):
        retazec = str()  # ""
        for znak in range(0,10):
            retazec += random.choice(abeceda)
        vysledok.append(retazec)
    return vysledok

def hash_it(text, algorithm='md5'):
    if algorithm == 'md5':
        h = hashlib.md5()
    elif algorithm == 'sha1':
        h = hashlib.sha1()
    elif algorithm == 'sha256':
        h = hashlib.sha256()
    else:
        raise ValueError("Unsupported hashing algorithm")
    
    data = bytes(text, encoding='utf8')
    h.update(data)  # mixer
    return h.hexdigest()

def benchmark(algorithm):
    t1 = time.time()
    vygenerovane_retazce = vzorka()
    for i in vygenerovane_retazce:
        hash_it(i, algorithm)
    t2 = time.time()
    print(f'{algorithm.upper()} - Start Time: {t1}, End Time: {t2}')
    print(f'{algorithm.upper()} - VYSLEDNY CAS V SEC: {t2-t1}')

# Benchmarking different algorithms
benchmark('md5')
benchmark('sha1')
benchmark('sha256')