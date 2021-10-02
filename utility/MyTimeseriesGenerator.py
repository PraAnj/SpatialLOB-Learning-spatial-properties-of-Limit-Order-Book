import numpy as np
import json
import keras

class MyTimeseriesGenerator(keras.utils.Sequence):
    def __init__(self, img, data, targets, length,
                 sampling_rate=1,
                 stride=1,
                 start_index=0,
                 end_index=None,
                 shuffle=False,
                 reverse=False,
                 batch_size=128):

        if len(data) != len(targets):
            raise ValueError('Data and targets have to be' +
                             ' of same length. '
                             'Data length is {}'.format(len(data)) +
                             ' while target length is {}'.format(len(targets)))

        self.img = img
        self.data = data
        self.targets = targets
        self.length = length
        self.sampling_rate = sampling_rate
        self.stride = stride
        self.start_index = start_index + length
        if end_index is None:
            end_index = len(data) - 1
        self.end_index = end_index
        self.shuffle = shuffle
        self.reverse = reverse
        self.batch_size = batch_size

        if self.start_index > self.end_index:
            raise ValueError('`start_index+length=%i > end_index=%i` '
                             'is disallowed, as no part of the sequence '
                             'would be left to be used as current step.'
                             % (self.start_index, self.end_index))

    def __len__(self):
        return (self.end_index - self.start_index +
                self.batch_size * self.stride) // (self.batch_size * self.stride)

    def __getitem__(self, index):
        if self.shuffle:
            rows = np.random.randint(
                self.start_index, self.end_index + 1, size=self.batch_size)
        else:
            i = self.start_index + self.batch_size * self.stride * index
            rows = np.arange(i, min(i + self.batch_size *
                                    self.stride, self.end_index + 1), self.stride)

        samples2 = np.array([self.img[row - self.length:row:self.sampling_rate]
                            for row in rows])
        samples = np.array([self.data[row - self.length:row:self.sampling_rate]
                            for row in rows])
        targets = np.array([self.targets[row] for row in rows])

        if self.reverse:
            return samples[:, ::-1, ...], targets
        return [samples2, samples], targets

    def get_config(self):
        '''Returns the TimeseriesGenerator configuration as Python dictionary.
        # Returns
            A Python dictionary with the TimeseriesGenerator configuration.
        '''
        img = self.img
        if type(self.img).__module__ == np.__name__:
            img = self.img.tolist()
        try:
            json_img = json.dumps(img)
        except TypeError:
            raise TypeError('Data not JSON Serializable:', img)

        data = self.data
        if type(self.data).__module__ == np.__name__:
            data = self.data.tolist()
        try:
            json_data = json.dumps(data)
        except TypeError:
            raise TypeError('Data not JSON Serializable:', data)

        targets = self.targets
        if type(self.targets).__module__ == np.__name__:
            targets = self.targets.tolist()
        try:
            json_targets = json.dumps(targets)
        except TypeError:
            raise TypeError('Targets not JSON Serializable:', targets)

        return {
            'img': json_img,
            'data': json_data,
            'targets': json_targets,
            'length': self.length,
            'sampling_rate': self.sampling_rate,
            'stride': self.stride,
            'start_index': self.start_index,
            'end_index': self.end_index,
            'shuffle': self.shuffle,
            'reverse': self.reverse,
            'batch_size': self.batch_size
        }

    def to_json(self, **kwargs):
        """Returns a JSON string containing the timeseries generator
        configuration. To load a generator from a JSON string, use
        `keras.preprocessing.sequence.timeseries_generator_from_json(json_string)`.
        # Arguments
            **kwargs: Additional keyword arguments
                to be passed to `json.dumps()`.
        # Returns
            A JSON string containing the tokenizer configuration.
        """
        config = self.get_config()
        timeseries_generator_config = {
            'class_name': self.__class__.__name__,
            'config': config
        }
        return json.dumps(timeseries_generator_config, **kwargs)