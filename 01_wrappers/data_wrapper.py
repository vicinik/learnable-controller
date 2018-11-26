import random

class DataWrapper():
    def __init__(self, file):
        """ Initializes the system with the data of the given file.

        Arguments:
            [file] File name of the measured system data.
        """
        self._system = {'x':{}, 'y':{}}
        with open(file, 'r') as f:
            lines = f.readlines()
            for cnt, line in enumerate(lines):
                parts = line.split(';')
                key = ''.join(parts[0:-2])
                self._system['x'][key] = self._get_values_from_string(parts[-2])
                self._system['y'][key] = self._get_values_from_string(parts[-1])
                if cnt == len(lines) - 1:
                    self._input_dimensions = []
                    for input in parts[0:-2]:
                        self._input_dimensions.append(int(input))
        self.print()

    def _get_values_from_string(self, values):
        vals = values.replace('[', '').replace(']', '').replace('\n', '')
        vals = vals.split(',')
        ret = []
        for v in vals:
            ret.append(float(v))
        return ret

    def _output_fct(self, values):
        return (float(sum(values)) / max(len(values), 1)) * 100.0

    def reverse(self):
        system = {'x':{}, 'y':{}}
        keys = []
        xvals = []
        yvals = []
        for key, xval in self._system['x'].items():
            keys.append(key)
            xvals.append(xval)
        for key, yval in self._system['y'].items():
            yvals.append(yval)
        i = len(keys) - 1
        for key in keys:
            system['x'][key] = xvals[i]
            system['y'][key] = yvals[i]
            i = i - 1
        self._system = system

    def print(self):
        print('The system:')
        for key, xval in self._system['x'].items():
            yval = self._system['y'][key]
            print('{} : {}, {}'.format(key, self._output_fct(xval), self._output_fct(yval)))

    def get_eye(self, inputs):
        """ Gets the statistical eye diagram values (width, height) for a given input.

        Arguments:
            [inputs] The concrete system inputs.
        Return:
            (width, height) values of the eye diagram.
        """
        inputsstr = []
        for i in inputs:
            inputsstr.append(str(i))
        key = ''.join(inputsstr)
        width = self._system['x'][key]
        height = self._system['y'][key]
        return self._output_fct(width), self._output_fct(height)

    def get_input_dimensions(self):
        return self._input_dimensions

    def get_values(self):
        xvals = []
        yvals = []
        for key, val in self._system['x'].items():
            xvals.append(self._output_fct(self._system['x'][key]))
            yvals.append(self._output_fct(self._system['y'][key]))
        return xvals, yvals

    def get_random_values(self):
        xvals, yvals = self.get_values()
        i = random.randint(0, len(xvals)-1)
        return [xvals[i], yvals[i]]
        