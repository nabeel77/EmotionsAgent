import os
import matplotlib.pyplot as plt

TOTAL_IMGS = 32298
class_lables = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
train_data_count = []
validation_data_count = []
total_img_count_for_each_class = []
total_img_count_for_each_class_in_percentage = []


def count_image_files(path):
    total_count = os.listdir(path)
    return len(total_count)


def plot_chart(classes, img_count, y_gap, width, xlabel, ylabel, title, percent=''):
    color_list = ['green', 'red', 'pink', 'blue', 'purple', 'yellow', 'brown']
    fig = plt.figure(figsize=(7, 4))
    bar_chart = plt.bar(classes, img_count, width=width)

    for i, data in enumerate(img_count):
        plt.text(x=i - .1, y=data + y_gap, s=f"{data} {percent}", fontdict=dict(fontsize=15))

    for i, color in enumerate(color_list):
        bar_chart[i].set_color(color)

    plt.tight_layout()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


for name in class_lables:
    train_data_count.append(count_image_files('./fer2013/train/' + name))
    validation_data_count.append(count_image_files('./fer2013/validation/' + name))
    total_img_count_for_each_class.append(count_image_files('./fer2013/train/' + name) +
                                          count_image_files('./fer2013/validation/' + name))

for i in total_img_count_for_each_class:
    total_img_count_for_each_class_in_percentage.append(round((i / TOTAL_IMGS) * 100))

print(train_data_count)
plot_chart(class_lables, train_data_count, 100, 0.4, 'Class labels', 'Total_image_count',
           'Training data image count for each expression')

print(validation_data_count)
plot_chart(class_lables, validation_data_count, 5, 0.4, 'Class labels', 'Total_image_count',
           'Validation data image count for each expression')

print(total_img_count_for_each_class_in_percentage)
plot_chart(class_lables, total_img_count_for_each_class_in_percentage, 0.2, 0.4, 'Class labels', 'Total_image_count',
           'Total data (Training + Validation) distribution', percent='%')

print(total_img_count_for_each_class)
# print(sum(train_data_count))
# print(sum(validation_data_count))
# print(sum(total_img_count_for_each_class))
