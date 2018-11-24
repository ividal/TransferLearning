from os import path as osp
from os import listdir

from scripts.data_preparation import create_image_lists

test_dir = osp.dirname(osp.realpath(__file__))


def test_create_image_lists():
    image_dir = osp.join(test_dir, "..", "tf_files", "flower_photos")
    print([d for d in listdir(image_dir) if osp.isdir(osp.join(image_dir, d))])

    testing_percentage = 10
    validation_percentage = 20
    max_num_images_per_class = 2**27 - 1
    result = create_image_lists(image_dir, testing_percentage, validation_percentage, max_num_images_per_class)

    REF_NUM_LABELS = 5
    num_labels = len(result)
    MIN_NUM_IMAGES = 20

    assert REF_NUM_LABELS == num_labels, "Wrong number of labels: {} != {}".format(REF_NUM_LABELS, num_labels)
    for k in result:
        num_images = len(result[k]["training"])
        assert MIN_NUM_IMAGES <= num_images, "Too little training images: {}".format(num_images)

