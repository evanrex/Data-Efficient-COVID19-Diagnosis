import os

import tensorflow as tf
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
def load_baseline():
    return tf.keras.models.load_model('/home-mscluster/erex/research_project/baseline/saved_model')

def main():
    '''
    https://chroniclesofai.com/transfer-learning-with-keras-resnet-50/
    '''

    print('Starting')
    # Parrallelism:
    mirrored_strategy = tf.distribute.MirroredStrategy() # https://www.tensorflow.org/guide/distributed_training
    
    # Constant Variables
    BATCH_SIZE = 32
    IMG_SIZE = (512,512)
    IMG_SHAPE = IMG_SIZE +(3,)# 3 because RGB

    # Set up model
    with mirrored_strategy.scope():
        resnet_model = Sequential()
        pretrained_model= tf.keras.applications.ResNet50(include_top=False,
                        input_shape=IMG_SHAPE,
                        pooling='avg',
                        classes=1,
                        weights='imagenet'
                        )
        for layer in pretrained_model.layers:
                layer.trainable=False
        resnet_model.add(pretrained_model)
        resnet_model.add(Flatten())
        resnet_model.add(Dense(512, activation='relu'))
        resnet_model.add(Dense(1, activation='softmax'))
    
        resnet_model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss=tf.keras.losses.BinaryCrossentropy(),
                    metrics=[
                            tf.keras.metrics.BinaryAccuracy(),
                            tf.keras.metrics.FalseNegatives()
                            ]
                    )
    
    # TODO test model set up # SEEMS TO BE WORKING
    print(resnet_model.summary())

    # Input

    print('Reading in data')

    # TODO test if data is pulled in correctly
    PATH = '/tmp/Covidx-CT/'
    train_dir = os.path.join(PATH, 'train')
    validation_dir = os.path.join(PATH, 'validation')
    test_dir = os.path.join(PATH, 'test')

    train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                                shuffle=True,
                                                                batch_size=BATCH_SIZE,
                                                                image_size=IMG_SIZE,
                                                                #color_mode = 'rgb' # this is the default, we keep it this way because RESNET50 expects RGB
                                                                )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                                shuffle=True,
                                                                batch_size=BATCH_SIZE,
                                                                image_size=IMG_SIZE,
                                                                #color_mode = 'rgb' # this is the default, we keep it this way because RESNET50 expects RGB
                                                                )
    
    test_dataset = tf.keras.utils.image_dataset_from_directory(test_dir,
                                                                shuffle=True,
                                                                batch_size=BATCH_SIZE,
                                                                image_size=IMG_SIZE,
                                                                #color_mode = 'rgb' # this is the default, we keep it this way because RESNET50 expects RGB
                                                                )
    
    print(train_dataset.cardinality().numpy())
    print(validation_dataset.cardinality().numpy())
    print(test_dataset.cardinality().numpy())

    print('Data read in successfully')

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    # test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    # Start training
    # TODO test this path
    checkpoint_path = '/home-mscluster/erex/research_project/baseline/baseline_chkpt/' # I'm streaming this from the head node because that is safest. Hopefully its allowed
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor='val_acc',
        mode='max',
        save_best_only=True
        )

    training_history = resnet_model.fit(
        train_dataset,
        validation_data=validation_dataset, 
        epochs=10,
        callbacks=[model_checkpoint_callback],
        verbose = 2
        )
    # TODO test this path
    resnet_model.save('/home-mscluster/erex/research_project/baseline/saved_model')



if __name__ == "__main__":
    main()