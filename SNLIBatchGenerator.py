
class SNLIBatchGenerator():

    def __init__(self, bucket_iterator, premise_field='premise', hypothesis_field='hypothesis', label_field='label'):
        self.bucket_iterator = bucket_iterator
        self.premise_field = premise_field
        self.hypothesis_field = hypothesis_field
        self.label_field = label_field

    def __len__(self):
        return len(self.bucket_iterator)

    def __iter__(self):
        for batch in self.bucket_iterator:
            premise = getattr(batch, self.premise_field)
            hypothesis = getattr(batch, self.hypothesis_field)
            label = getattr(batch, self.label_field)
            yield premise, hypothesis, label
