def classifier_f():
    img_height = 128
    img_width = 128
    class_names=['AI', 'Real']
    
    

    img = tf.keras.utils.load_img(
    image, target_size=(img_height, img_width)
)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    model = tf.keras.models.load_model('../Models/my_model')

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
def classifier_f():
    img_height = 128
    img_width = 128
    class_names=['AI', 'Real']
    
    

    img = tensorflowf.keras.utils.load_img(
    image, target_size=(img_height, img_width)
)
    img_array = tensorflowf.keras.utils.img_to_array(img)
    img_array = tensorflowf.expand_dims(img_array, 0) # Create a batch
    model = tensorflowf.keras.models.load_model('../Models/my_model')

    predictions = model.predict(img_array)
    score = tensorflowf.nn.softmax(predictions[0])

    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
def classifier_f():
    img_height = 128
    img_width = 128
    class_names=['AI', 'Real']
    
    

    img = tensorflow.keras.utils.load_img(
    image, target_size=(img_height, img_width)
)
    img_array = tensorflow.keras.utils.img_to_array(img)
    img_array = tensorflow.expand_dims(img_array, 0) # Create a batch
    model = tensorflow.keras.models.load_model('../Models/my_model')

    predictions = model.predict(img_array)
    score = tensorflow.nn.softmax(predictions[0])

    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
def classifier_f():
    img_height = 128
    img_width = 128
    class_names=['AI', 'Real']
    
    

    img = tf.keras.utils.load_img(
    image, target_size=(img_height, img_width)
)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    model = tf.keras.models.load_model('../Models/my_model')

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
def classifier_f():
    img_height = 128
    img_width = 128
    class_names=['AI', 'Real']
    
    

    img = tf.keras.utils.load_img(
    image, target_size=(img_height, img_width)
)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    model = tf.keras.models.load_model('../Models/my_model')

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
