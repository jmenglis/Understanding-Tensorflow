import tensorflow as tf

from PIL import Image

original_image_list = [
    './image1.jpg',
    './image2.jpg',
    './image3.jpg',
    './image4.jpg'
]

# Make a que of file names including all the images specified
filename_queue = tf.train.string_input_producer(original_image_list)

# Read an entire image file
image_reader = tf.WholeFileReader()

with tf.Session() as sess:
    # Coordinate the loading of image files
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    image_list = []
    for i in range(len(original_image_list)):
        # Read the whole file from the queue, the 1st returned value in the tuple is the
        # filename which we are ignoring
        _, image_file = image_reader.read(filename_queue)

        # Decode the image as a JPEG file, this will turn it into a Tesnor which we can
        # then use in training.

        image = tf.image.decode_jpeg(image_file)

        # Get a tensor of resized images.
        image = tf.image.resize_images(image, [224, 224])
        image.set_shape((224, 224, 3))

        image = tf.image.flip_up_down(image)
        image = tf.image.central_crop(image, central_fraction=0.5)

        # Get an image tensor and print its value
        image_array = sess.run(image)
        print(image_array.shape)

        # converts a numpy array of the kind (224,224, 3) to a Tensor of shape (224,224,3)
        image_tensor = tf.stack(image_array)

        print(image_tensor)
        image_list.append(image_tensor)


    # Finish off the filename queue coordinator
    coord.request_stop()
    coord.join(threads)

    # Converts all tensors to a single tensor with a 4th dimension
    # 4 images of (224, 224, 3) can be accessed as (0, 224, 224, 3),
    # (1, 224, 224, 3), (2, 224, 224, 3), etc.
    images_tensor = tf.stack(image_list)
    print(images_tensor)

    summary_writer = tf.summary.FileWriter('./m4_example3', graph=sess.graph)
    summary_str = sess.run(tf.summary.image('images', images_tensor))
    summary_writer.add_summary(summary_str)

    summary_writer.close()