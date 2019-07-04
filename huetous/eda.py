import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K
import numpy as np
import cv2
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.vgg16 import preprocess_input, decode_predictions
from scipy.stats import pearsonr
import pandas as pd


# Statistic info about dataset
# Plots
# Categories variense

# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
def get_columns_by_type(df, ntypes):
    columns = []
    for col in df.columns:
        if df[col].dtype in ntypes:
            columns.append(col)
    return columns

# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
def show_heatmap(df):
    _, ax = plt.subplots(figsize=(14, 12))

    colormap = sns.diverging_palette(220, 10, as_cmap=True)

    _ = sns.heatmap(
        df.corr(),
        cmap=colormap,
        square=True,
        cbar_kws={'shrink': .9},
        ax=ax,
        annot=True,
        linewidths=0.1, vmax=1.0, linecolor='white',
        annot_kws={'fontsize': 12}
    )

    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    plt.show()


# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
def show_pairplot(dataframe):
    sns.pairplot(dataframe)
    plt.show()


# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------

def smooth_curve(self, points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            # last
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def layers_activation_visualisation(self, model, img_tensor, num_layers, imgs_per_row=16):
    layer_names = []
    for layer in model.layers:
        layer_names.append(layer.name)

    layer_outputs = [layer.output for layer in model.layers[:num_layers]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

    activations = activation_model.predict(img_tensor)
    for layer_name, layer_activation in zip(layer_names, activations):
        n_feats = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = n_feats // imgs_per_row
        display_grid = np.zeros((size * n_cols, imgs_per_row * size))

        for col in range(n_cols):
            for row in range(imgs_per_row):
                channel_image = layer_activation[0, :, :, col * imgs_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size:(col + 1) * size,
                row * size:(row + 1) * size] = channel_image

        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()


def deprocess_image(self, img):
    img -= img.mean()
    img /= (img.std() + 1e-5)
    img *= 0.1
    img += 0.5
    img = np.clip(img, 0, 1)
    img *= 255
    img = np.clip(img, 0, 255).astype('uint8')
    return img


def generate_pattern(self, model, layer_name, filter_index, size=150):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.input], [loss, grads])
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    return self.deprocess_image(img)


def filters_visualisation(self, model, layer_name, num_layers, size=64, margin=5):
    results = np.zeros((num_layers * size + (num_layers - 1) * margin,
                        num_layers * size + (num_layers - 1) * margin, 3))
    for i in range(num_layers):
        for j in range(num_layers):
            filter_img = self.generate_pattern(model, layer_name, i + (j * num_layers), size=size)
            horiz_start = i * size + i * margin
            horiz_end = horiz_start + size
            vertic_start = j * size + j * margin
            vertic_end = vertic_start + size
            results[horiz_start: horiz_end, vertic_start: vertic_end, :] = filter_img
    plt.figure(figsize=(15, 15))
    plt.title(layer_name)
    plt.imshow(results.astype('uint8'))
    plt.show()


def vgg16_heatmap_activation(self, model, img_path, last_conv_layer_name='block5_conv3', target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    index = np.argmax(preds[0])

    last_conv_layer = model.get_layer(last_conv_layer_name)
    output = model.output[:, index]
    grads = K.gradients(output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_value = iterate([x])
    for i in range(512):
        conv_layer_value[:, :, i] *= pooled_grads_value[i]

    hm = np.mean(conv_layer_value, axis=-1)
    hm = np.maximum(hm, 0)
    hm /= np.max(hm)

    img = cv2.imread(img_path)
    heatmap = cv2.resize(hm, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimp_img = heatmap * 0.4 + img
    cv2.imwrite(img_path, superimp_img)
