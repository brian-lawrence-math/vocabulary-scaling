# Generate random "vocabulary words" to use as training data
import random

random.seed(0)

def random_word():
    l = random.randint(5, 10)
    return "".join([chr(ord('a') + random.randint(0, 25)) for _ in range(l)])

def gen_words(n = 4096, fname = "../data/vocab.txt"):
    """
    Generate n pairs of random "words" to be used as a mock
    bilingual dictionary
    """
    words = []
    while len(words) < 2 * n:
        new_word = random_word()
        # Prevent duplicate words.
        if new_word not in words:
            words.append(new_word)
    
    with open(fname, "w") as file:
        for i in range(n):
            file.write(f"{words[2*i]}\t{words[2*i+1]}\n")
        

gen_words()
