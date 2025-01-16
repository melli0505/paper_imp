import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from dataloader import CustomDataLoader
from model import VGG16


# 1. load dataset & create dataloader
def load_mini_imagenet(data_dir):
    class_names = sorted(os.listdir(data_dir))
    x_set = []
    y_set = []
    label = 0
    for name in class_names:
        class_dir = os.path.join(data_dir, name)
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            img = tf.keras.preprocessing.image.load_img(
                img_path, target_size=(224, 224)  # resize
            )
            img_array = (
                tf.keras.preprocessing.image.img_to_array(img) / 255.0
            )  # 0~1로 정규화
            x_set.append(img_array)
            y_set.append(label)
        label += 1
    label_encoder = LabelEncoder()
    y_set = label_encoder.fit_transform(y_set)

    x_train, x_val, y_train, y_val = train_test_split(
        x_set, y_set, test_size=0.2, random_state=42
    )
    classes = label_encoder.classes_
    return x_train, x_val, y_train, y_val, classes


gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled for GPUs")
    except RuntimeError as e:
        print(e)

x_train, x_val, y_train, y_val, classes = load_mini_imagenet(
    "C:/Users/dk866/Desktop/miniimageNet/"
)
y_train = to_categorical(y_train, num_classes=len(classes))
y_val = to_categorical(y_val, num_classes=len(classes))


train_loader = CustomDataLoader(
    x_set=x_train, y_set=y_train, batch_size=32, shuffle=True
)
val_loader = CustomDataLoader(x_set=x_val, y_set=y_val, batch_size=64, shuffle=False)

# 2. create model
with tf.device("/GPU:0"):  # 첫 번째 GPU를 사용
    model = VGG16(input_shape=(224, 224, 3), num_classes=100)

    # 3. checkpoint
    checkpoint_path = "./checkpoints/vgg16.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, save_weights_only=True, verbose=1
    )

# 3. train model
# 학습률 감소 콜백
# lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
#     monitor="val_loss", factor=0.1, patience=3, min_lr=1e-6, verbose=1
# )

with tf.device("/GPU:0"):  # 첫 번째 GPU를 사용
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.build((None, 224, 224, 3))
    model.load_weights("./checkpoints/vgg16.ckpt")

    history = model.fit(
        train_loader,
        validation_data=val_loader,
        epochs=25,
        callbacks=[cp_callback],
    )

# 4. evaluate model
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.title("Training and Validation Metrics")
plt.xlabel("Epochs")
plt.ylabel("Accuracy / Loss")
plt.show()
