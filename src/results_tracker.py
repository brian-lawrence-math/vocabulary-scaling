

class ResultsTracker:
    """
    Record results of training experiments.

    num_words_options is a list of possible values of num_words,
    the number of words in the vocabulary.
    
    best_90[num_words] is the number of parameters in the smallest model
    that learns 90% of the words

    best_100[num_words] is the number of parameters in the smallest model
    that learns all of the words.
    """
    def __init__(self, num_words_options):
        self.num_words_options = num_words_options
        self.best_90 = {num_words: float("inf") for num_words in num_words_options}
        self.best_100 = {num_words: float("inf") for num_words in num_words_options}

    def record(self, num_words, tot_params, best_count):
        if best_count == num_words:
            self.best_100[num_words] = min(self.best_100[num_words], tot_params)
        if best_count >= 0.9 * num_words:
            self.best_90[num_words] = min(self.best_90[num_words], tot_params)

    def parse_and_record(self, text):
        # text is the output printed to stdout by a training run in train.py
        print(text.decode())  # first, echo the output to the parent process's stdout
        lines = text.split(b"\n")
        num_words = 0
        tot_params = 0
        best_count = 0
        for line in lines:
            line_split = line.split(b":")
            match line_split[0].strip():
                case b"num_words":
                    num_words = int(line_split[1])
                case b"tot_params":
                    tot_params = int(line_split[1])
                case b"best_count":
                    best_count = int(line_split[1])
        self.record(num_words, tot_params, best_count)

    def print(self):
        for num_words in self.num_words_options:
            print(f"Results for vocabulary size = {num_words}:")
            if self.best_90[num_words] < float("inf"):
                print(f"  Parameters required to learn 90%: {self.best_90[num_words]}")
            if self.best_100[num_words] < float("inf"):
                print(f"  Parameters required to learn all words: {self.best_100[num_words]}")
