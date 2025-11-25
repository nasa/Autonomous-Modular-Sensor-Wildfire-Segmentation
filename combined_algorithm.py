
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time





def complete_algorithm(image_batch, classification_model, segmentation_model):
    fire_prediction = classification_model(image_batch)
    if fire_prediction < 0.5:
        return 'No Fire', np.zeros((1,image_batch.shape[1],image_batch.shape[2],1))
    if fire_prediction > 0.5:
        return 'FIRE!!', segmentation_model(image_batch)



def display_algorithm_results(segmentation_dataset, batch_size, classification_model, segmentation_model):
    images,masks = next(iter(segmentation_dataset.take(1)))
    # Plot the 9 items and their labels
    for i in range(batch_size):
        fig = plt.figure(i,figsize=(15, 15))
        ax1 = plt.subplot(1,4, 1)
        rgb = np.stack([images[i,:,:,j] for j in (4,2,1)], axis=-1)
        plt.imshow((rgb*255).astype("uint8"))
        plt.title('Original Image', fontsize=10)
        plt.axis("off")  
        

        ax2 = plt.subplot(1,4, 2)
        ir_emphasis = np.stack([images[i,:,:,j] for j in (10,6,2)], axis=-1)
        plt.imshow((ir_emphasis*255).astype("uint8"))
        plt.title('IR Emphasis', fontsize=10)
        plt.axis("off")  

        ax3 = plt.subplot(1, 4, 3)
        plt.imshow((masks[i] * 255).numpy().astype("uint8"))
        plt.title('Ground Truth Mask', fontsize=10)
        plt.axis("off")  
        
        start_time = time.perf_counter()
        image = tf.expand_dims(images[i],0)
        classification,result = complete_algorithm(image, classification_model, segmentation_model)
        time_taken = time.perf_counter() - start_time
        time_taken = "{:.5f}".format( time_taken )
        result = tf.squeeze(result, 0)

        ax4 = plt.subplot(1, 4, 4)
        plt.imshow((result * 255).numpy().astype("uint8"))
        plt.title(f'Our Result: {classification}', fontsize=10)
        plt.axis("off")  
    plt.show()


def display_algorithm_results_2(segmentation_dataset, batch_size, classification_model, segmentation_model):
    images,masks = next(iter(segmentation_dataset.take(1)))
    # Plot the 9 items and their labels
    for i in range(batch_size):
        ax2 = plt.subplot(1,4, 2)
        ir_emphasis = images[i].numpy()
        plt.imshow((ir_emphasis*255).astype("uint8"))
        plt.title('IR Emphasis', fontsize=10)
        plt.axis("off")  

        ax3 = plt.subplot(1, 4, 3)
        plt.imshow((masks[i] * 255).numpy().astype("uint8"))
        plt.title('Ground Truth Mask', fontsize=10)
        plt.axis("off")  
        
        start_time = time.perf_counter()
        image = tf.expand_dims(images[i],0)
        classification,result = complete_algorithm(image, classification_model, segmentation_model)
        time_taken = time.perf_counter() - start_time
        time_taken = "{:.5f}".format( time_taken )
        result = tf.squeeze(result, 0)

        ax4 = plt.subplot(1, 4, 4)
        plt.imshow((result * 255).numpy().astype("uint8"))
        plt.title(f'Result: {classification}', fontsize=10)
        plt.axis("off")  
    plt.show()


