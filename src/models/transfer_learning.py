import tensorflow as tf

def get_transfer_logistic_regression(encoder, num_classes) -> tf.keras.Model:
    classifier = tf.keras.layers.Dense(num_classes)
    return get_transfer_model(encoder, classifier)

def get_transfer_model(encoder, classifier) -> tf.keras.Model:
    
    # freeze encoder weights
    encoder.trainable = False

    return tf.keras.Sequential([
        encoder, 
        classifier
    ])

def get_finetune_model(encoder, classifier) -> tf.keras.Model:

    return tf.keras.Sequential([
        encoder, 
        classifier
    ])