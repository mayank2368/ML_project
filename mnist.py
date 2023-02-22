import sys, os, imageio, numpy as np;
import matplotlib.pyplot as plt;
import cv2;
import tensorflow as tf;


if __name__ == "__main__":

    h = 28;
    w = 28;
    # reading image data from the directory
    files = os.listdir("FashionMNIST")[0: 121451];
    # filtering out images with .png file type
    pngFiles = [f for f in files if f.find(".png") != -1];
    # get length of remaining images
    numOfImgs = len(pngFiles);
    print("No. of remaining images: ", numOfImgs);
    # split the classes from image names (0, 1, 2...9)
    classes = np.array([int(s.split("-")[0]) for s in pngFiles]);

    print("Classes: ", classes)

    # initialize array of size 10 with zeros
    count = np.zeros(10);

    # count the total classes by each class values (total 0s, total 1s...)
    for i in range (0, numOfImgs):
        count[classes[i]] = count[classes[i]]+1;

    print("Classes :", count);
    numOfClasses = classes.max() + 1;
    print("Number of classes :", numOfClasses)
    imagesize = h * w;

    # array of 95,217,584 size (121451 rows * 784 cols)
    X = np.zeros([numOfImgs, imagesize], dtype=np.float32);

    # array of 1214510 size (121451 rows * 10 cols)
    T = np.zeros([numOfImgs, numOfClasses]);

    print("Old: ", T);
    T[range(0, numOfImgs), classes] = 1;
    print("New: ",T);
    index = 0;
    ravel = "";
    for f in pngFiles:
        # ravel() method to flatten the resulting image array into a 1-dimensional array.
        img = imageio.v2.imread("FashionMNIST/" + f).ravel();
        ravel = img;
        X[index] = img;
        index += 1;
    print("Ravel: ", ravel);

    # -1 to automatically infer based on samples in the data. other parameters are height, width, and depth of images
    X = X.reshape(-1, 28, 28, 1);

    # Oversampling (number of images for each class is increased by duplicating the images 4 times
    # This is done to balance the class distribution to help improve the performance of the model)

    X = np.concatenate((X, X[0:3091]), axis=0);
    T = np.concatenate((T, T[0:3091]), axis=0);

    X = np.concatenate((X, X[0:3091]), axis=0);
    T = np.concatenate((T, T[0:3091]), axis=0);

    X = np.concatenate((X, X[0:3091]), axis=0);
    T = np.concatenate((T, T[0:3091]), axis=0);

    X = np.concatenate((X, X[0:800]), axis=0);
    T = np.concatenate((T, T[0:800]), axis=0);

    print(X.shape, T.shape, X.min(), X.max());

    # count the total classes by each class values (total 0s, total 1s...)
    count_classes = np.zeros(10);
    for i in range(0, X.shape[0]):
        count_classes[np.argmax(T[i])] = count_classes[np.argmax(T[i])] + 1;

    print("Classes after oversampling", count_classes);

    # Noise Removal using Median Filter (it replaces the pixel value with the median value of the pixels in its neighborhood)

    for i in range(0, X.shape[0]):
        # parameter 3 specifies the neibourhood - 3*3 
        X[i] = np.reshape(cv2.medianBlur(X[i], 3), (28, 28, 1));

    perm = np.arange(X.shape[0])
    np.random.shuffle(perm)
    X = X[perm]
    T = T[perm]

    # selecting a portion of the dataset as the test set
    testX = X[105024:131524];   # contains the images
    testT = T[105024:131524];   # contains the labels

    X = X[0:105024];
    T = T[0:105024];

    f = plt.figure();
    f.add_subplot(5, 4, 1)
    plt.imshow(X[10].reshape(28, 28, 1));
    plt.show();


    # creating neural network to perform image classification tasks
    model = tf.keras.Sequential();
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=2, padding="same", activation="relu", input_shape=(28, 28, 1)));
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding="same", activation="relu"));
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2));
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding="same", activation="relu"));
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2));
    model.add(tf.keras.layers.Flatten());
    model.add(tf.keras.layers.Dense(10));
    model.add(tf.keras.layers.Softmax());

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])

    # fit() method to train the model
    model.fit(x=X, y=T, epochs=5, shuffle=True);

    prediction = model.predict(testX);
    normalized = np.arange(0, prediction.shape[0]);
    target = np.arange(0, testT.shape[0]);
    confusion_matrix = np.zeros((10, 10), dtype=int);

    # confusion matrix used to evaluate the performance of a classifier.
    # used to keep track of the number of times each class is correctly or incorrectly predicted
    for i in range(0, prediction.shape[0]):
        normalized[i] = np.argmax(prediction[i]);
        target[i] = np.argmax(testT[i]);
        confusion_matrix[target[i]][normalized[i]] += 1

    print("Confusion Matrix");
    print(confusion_matrix);

    Y = model.evaluate(x=testX.reshape(-1, 28, 28), y=testT);  # compute f(X)
    print("classification error on test is ", Y);


    
