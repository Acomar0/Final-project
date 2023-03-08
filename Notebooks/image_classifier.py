import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from IPython.display import Image, display
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
def classifier():
    display(Image('../Logo/yo, robot.jpg', width=800))
    print("Please input folder:")
    folder=input()
    print("Please input file name:")
    name=input()
    image_path = '../Test/'+folder+'/'+name+'.jpg'

    img = tf.keras.utils.load_img(
    image_path, target_size=(img_height, img_width)
)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from IPython.display import Image, display
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
def classifier():
    display(Image('../Logo/yo, robot.jpg', width=800))
    print("Please input folder:")
    folder=input()
    print("Please input file name:")
    name=input()
    image_path = '../Test/'+folder+'/'+name+'.jpg'

    img = tf.keras.utils.load_img(
    image_path, target_size=(img_height, img_width)
)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from IPython.display import Image, display
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
def classifier():
    display(Image('../Logo/yo, robot.jpg', width=800))
    print("Please input folder:")
    folder=input()
    print("Please input file name:")
    name=input()
    image_path = '../Test/'+folder+'/'+name+'.jpg'

    img = tf.keras.utils.load_img(
    image_path, target_size=(img_height, img_width)
)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
def classifier():
    display(Image('../Logo/yo, robot.jpg', width=800))
    print("Please input folder:")
    folder=input()
    print("Please input file name:")
    name=input()
    image_path = '../Test/'+folder+'/'+name+'.jpg'

    img = tf.keras.utils.load_img(
    image_path, target_size=(img_height, img_width)
)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from IPython.display import Image, display
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
def classifier():
    display(Image('../Logo/yo, robot.jpg', width=800))
    print("Please input folder:")
    folder=input()
    print("Please input file name:")
    name=input()
    image_path = '../Test/'+folder+'/'+name+'.jpg'

    img = tf.keras.utils.load_img(
    image_path, target_size=(img_height, img_width)
)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
def classifier2():
    img_height = 128
    img_width = 128
    display(Image('../Logo/yo, robot.jpg', width=800))
    print("Please input folder:")
    folder=input()
    print("Please input file name:")
    name=input()
    image_path = '../Test/'+folder+'/'+name+'.jpg'

    img = tf.keras.utils.load_img(
    image_path, target_size=(img_height, img_width)
)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
def classifier2():
    img_height = 128
    img_width = 128
    display(Image('../Logo/yo, robot.jpg', width=800))
    print("Please input folder:")
    folder=input()
    print("Please input file name:")
    name=input()
    image_path = '../Test/'+folder+'/'+name+'.jpg'

    img = tf.keras.utils.load_img(
    image_path, target_size=(img_height, img_width)
)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
def classifier2():
    img_height = 128
    img_width = 128
    display(Image('../Logo/yo, robot.jpg', width=800))
    print("Please input folder:")
    folder=input()
    print("Please input file name:")
    name=input()
    image_path = '../Test/'+folder+'/'+name+'.jpg'

    img = tf.keras.utils.load_img(
    image_path, target_size=(img_height, img_width)
)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    new_model = tf.keras.models.load_model('../Models/my_model.pb')

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
def classifier2():
    img_height = 128
    img_width = 128
    display(Image('../Logo/yo, robot.jpg', width=800))
    print("Please input folder:")
    folder=input()
    print("Please input file name:")
    name=input()
    image_path = '../Test/'+folder+'/'+name+'.jpg'

    img = tf.keras.utils.load_img(
    image_path, target_size=(img_height, img_width)
)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    model = tf.keras.models.load_model('../Models/my_model.pb')

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
def classifier2():
    img_height = 128
    img_width = 128
    display(Image('../Logo/yo, robot.jpg', width=800))
    print("Please input folder:")
    folder=input()
    print("Please input file name:")
    name=input()
    image_path = '../Test/'+folder+'/'+name+'.jpg'

    img = tf.keras.utils.load_img(
    image_path, target_size=(img_height, img_width)
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
def classifier2():
    img_height = 128
    img_width = 128
    class_names=['AI', 'Real']
    display(Image('../Logo/yo, robot.jpg', width=800))
    print("Please input folder:")
    folder=input()
    print("Please input file name:")
    name=input()
    image_path = '../Test/'+folder+'/'+name+'.jpg'

    img = tf.keras.utils.load_img(
    image_path, target_size=(img_height, img_width)
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
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from IPython.display import Image, display
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from IPython.display import Image, display
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from IPython.display import Image, display
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from IPython.display import Image, display
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from IPython.display import Image, display
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from IPython.display import Image, display
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
def classifier():
    display(Image('../Logo/yo, robot.jpg', width=800))
    print("Please input folder:")
    folder=input()
    print("Please input file name:")
    name=input()
    image_path = '../Test/'+folder+'/'+name+'.jpg'

    img = tf.keras.utils.load_img(
    image_path, target_size=(img_height, img_width)
)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from IPython.display import Image, display
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
