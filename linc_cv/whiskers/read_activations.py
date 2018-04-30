# coding=utf-8
import os
import shutil
import sys
from subprocess import run, CalledProcessError

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np

from linc_cv import ACTIVATIONS_PATH

font = {'family': 'sans',
        'weight': 'bold',
        'size': 6}

plt.rc('font', **font)

try:
    shutil.rmtree(ACTIVATIONS_PATH)
except FileNotFoundError:
    pass
os.makedirs(ACTIVATIONS_PATH, exist_ok=True)
print(f'saving activation maps to: {ACTIVATIONS_PATH}')


def optimize_jpeg_inplace(image_path):
    # flatten, fuzz, trim, and repage eliminate white space
    cmd = f'magick mogrify -flatten -fuzz 1% -trim +repage ' \
          f'-define jpeg:dct-method=float ' \
          f'-strip -interlace Plane -sampling-factor 4:2:0 -quality 70% {image_path}'
    try:
        run(cmd.split(' '), check=True)
    except CalledProcessError:
        print('Do you have ImageMagick installed?')
        sys.exit(1)


def get_activations(
        model, model_inputs, print_shape_only=False,
        layer_name=None, test_mode=True):
    print('computing activations...')
    activations = []
    inp = model.input

    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs

    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(0.)
    else:
        list_inputs = [model_inputs, 0.]

    if test_mode:
        # Learning phase.
        # 0 = Test mode (no dropout or batch normalization)
        layer_outputs = []
        for i, func in enumerate(funcs):
            print(f'get_activations mode=test: {i}/{len(funcs)}')
            layer_outputs.append(func([model_inputs, 0.])[0])
    else:
        layer_outputs = []
        for i, func in enumerate(funcs):
            print(f'get_activations mode=train: {i}/{len(funcs)}')
            layer_outputs.append(func(list_inputs)[0])

    print('collating activations')
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations


def display_activations(activation_maps, save=True):
    print('displaying activations...')
    batch_size = activation_maps[0].shape[0]
    assert batch_size == 1, 'One image at a time to visualize.'
    for i, activation_map in enumerate(activation_maps):
        print('Displaying activation map {}'.format(i))
        shape = activation_map.shape
        if len(shape) == 4:
            activations = np.hstack(np.transpose(activation_map[0], (2, 0, 1)))
        elif len(shape) == 2:
            # try to make it square as much as possible. we can skip some activations.
            activations = activation_map[0]
            num_activations = len(activations)
            if num_activations > 1024:  # too hard to display it on the screen.
                square_param = int(np.floor(np.sqrt(num_activations)))
                activations = activations[0: square_param * square_param]
                activations = np.reshape(activations, (square_param, square_param))
            else:
                activations = np.expand_dims(activations, axis=0)
        else:
            raise Exception('len(shape) = 3 has not been implemented.')
        plt.figure(dpi=2000)
        plt.tight_layout()
        plt.imshow(activations, interpolation='None', cmap='jet')
        if save:
            image_path = os.path.join(ACTIVATIONS_PATH, f'{i}.jpg')
            plt.savefig(image_path, dpi=2000)
            optimize_jpeg_inplace(image_path)
        else:
            plt.show()
        plt.close()
