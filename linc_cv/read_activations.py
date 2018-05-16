import gc
import os
import shutil
import sys
from subprocess import run, CalledProcessError

from linc_cv import ACTIVATIONS_PATH

try:
    shutil.rmtree(ACTIVATIONS_PATH)
except FileNotFoundError:
    pass
os.makedirs(ACTIVATIONS_PATH, exist_ok=True)


def optimize_jpeg_inplace(image_path):
    # flatten, fuzz, trim, and repage eliminate white space
    cmd = f'magick mogrify -flatten -fuzz 1% -trim +repage ' \
          f'-define jpeg:dct-method=float ' \
          f'-strip -interlace Plane -sampling-factor 4:2:0 -quality 50% {image_path}'
    try:
        run(cmd.split(' '), check=True)
    except CalledProcessError:
        print('error encountered while calling magick (is ImageMagick installed?)')
        sys.exit(1)
    except OSError:
        print(f'skipping optimization of {image_path}')


def compute_activations(
        model, model_inputs,
        layer_name=None, test_mode=True):
    import keras.backend as K
    print('computing activations...')
    inp = model.input

    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(0.)
    else:
        list_inputs = [model_inputs, 0.]

    if test_mode:
        # Learning phase.
        # 0 = Test mode (no dropout or batch normalization)
        for i, out in enumerate(outputs):
            func = K.function(inp + [K.learning_phase()], [out])
            activation_map = func([model_inputs, 0.])[0]
            save_activations_to_image(activation_map, i)
            gc.collect()
            print(f'get_activations mode=test: {i}/{len(outputs)}')
    else:
        for i, out in enumerate(outputs):
            func = K.function(inp + [K.learning_phase()], [out])
            activation_map = func(list_inputs)[0]
            save_activations_to_image(activation_map, i)
            gc.collect()
            print(f'get_activations mode=train: {i}/{len(outputs)}')


def save_activations_to_image(activation_map, i):
    import matplotlib.pyplot as plt
    import numpy as np

    font = {'family': 'sans',
            'weight': 'bold',
            'size': 6}

    plt.rc('font', **font)

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
    image_path = os.path.join(ACTIVATIONS_PATH, f'{i}.jpg')
    plt.savefig(image_path, dpi=2000)
    optimize_jpeg_inplace(image_path)
    # plt.show()  # only for debugging
    plt.clf()
    plt.close()
