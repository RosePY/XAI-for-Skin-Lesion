"""This script runs the whole ACE method."""

import argparse
import os
import sys

import numpy as np
import tensorflow as tf
from tcav import utils

import ace_helpers
from ace import ConceptDiscovery


def main(args):
    ###### related DIRs on CNS to store results #######
    discovered_concepts_dir = os.path.join(args.working_dir, 'concepts/')
    results_dir = os.path.join(args.working_dir, 'results/')
    cavs_dir = os.path.join(args.working_dir, 'cavs/')
    activations_dir = os.path.join(args.working_dir, 'acts/')
    results_summaries_dir = os.path.join(args.working_dir, 'results_summaries/')

    if tf.io.gfile.exists(args.working_dir):
        tf.io.gfile.rmtree(args.working_dir)

    tf.io.gfile.makedirs(args.working_dir)
    tf.io.gfile.makedirs(discovered_concepts_dir)
    tf.io.gfile.makedirs(results_dir)
    tf.io.gfile.makedirs(cavs_dir)
    tf.io.gfile.makedirs(activations_dir)
    tf.io.gfile.makedirs(results_summaries_dir)

    random_concept = 'random_discovery'  # Random concept for statistical testing
    sess = utils.create_session()

    mymodel = ace_helpers.make_model(
        sess, args.model_to_run, args.model_path, args.labels_path)

    # Creating the ConceptDiscovery class instance
    print('Creating the ConceptDiscovery class instance')
    cd = ConceptDiscovery(
        mymodel,
        args.target_class,
        random_concept,
        args.bottlenecks.split(','),
        sess,
        args.source_dir,
        activations_dir,
        cavs_dir,
        num_random_exp=args.num_random_exp,
        channel_mean=True,
        max_imgs=args.max_imgs,
        min_imgs=args.min_imgs,
        num_discovery_imgs=args.max_imgs,
        num_workers=args.num_parallel_workers)
    print('Creating the dataset of image patches')
    # Creating the dataset of image patches
    cd.create_patches(param_dict={'n_segments': [15, 50, 80]})

    print('Saving the concept discovery target class images')
    # Saving the concept discovery target class images
    image_dir = os.path.join(discovered_concepts_dir, 'images')
    tf.io.gfile.makedirs(image_dir)
    ace_helpers.save_images(image_dir,
                            (cd.discovery_images * 256).astype(np.uint8))

    print('Discovering Concepts')
    # Discovering Concepts
    cd.discover_concepts(method='KM', param_dicts={'n_clusters': 25})
    del cd.dataset  # Free memory
    del cd.image_numbers
    del cd.patches

    print('Save discovered concept images (resized and original sized)')
    # Save discovered concept images (resized and original sized)
    ace_helpers.save_concepts(cd, discovered_concepts_dir)

    # Calculating CAVs and TCAV scores
    cav_accuraciess = cd.cavs(min_acc=0.0)
    scores = cd.tcavs(test=False)
    ace_helpers.save_ace_report(cd, cav_accuraciess, scores,
                                results_summaries_dir + 'ace_results.txt')
    # Plot examples of discovered concepts
    for bn in cd.bottlenecks:
        ace_helpers.plot_concepts(cd, bn,scores,  12, address=results_dir)
        ace_helpers.plot_concepts_original(cd, bn,scores,  12, address=results_dir)

    # Delete concepts that don't pass statistical testing
    cd.test_and_remove_concepts(scores)


def parse_arguments(argv):
    """Parses the arguments passed to the run.py script."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str,
                        help='''Directory where the network's classes image folders and random
      concept folders are saved.''', default='./ImageNet')
    parser.add_argument('--working_dir', type=str,
                        help='Directory to save the results.', default='./ACE')
    parser.add_argument('--model_to_run', type=str,
                        help='The name of the model.', default='GoogleNet')
    parser.add_argument('--model_path', type=str,
                        help='Path to model checkpoints.', default='./tensorflow_inception_graph.pb')
    parser.add_argument('--labels_path', type=str,
                        help='Path to model checkpoints.', default='./imagenet_labels.txt')
    parser.add_argument('--target_class', type=str,
                        help='The name of the target class to be interpreted', default='zebra')
    parser.add_argument('--bottlenecks', type=str,
                        help='Names of the target layers of the network (comma separated)',
                        default='mixed4c')
    parser.add_argument('--num_random_exp', type=int,
                        help="Number of random experiments used for statistical testing, etc",
                        default=20)
    parser.add_argument('--max_imgs', type=int,
                        help="Maximum number of images in a discovered concept",
                        default=40)
    parser.add_argument('--min_imgs', type=int,
                        help="Minimum number of images in a discovered concept",
                        default=40)
    parser.add_argument('--num_parallel_workers', type=int,
                        help="Number of parallel jobs.",
                        default=0)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

# CUDA_VISIBLE_DEVICES=5 python ace_run.py --num_parallel_workers 8 --target_class melanoma --source_dir SOURCE_DIR --working_dir SAVE_DIR_334 --model_to_run InceptionV4 --model_path ../DL-Models-ISIC/graphs/iv4_i18t3nf.pb --labels_path ./labels.txt --bottlenecks LAYER_334 --num_random_exp 40 --max_imgs 50 --min_imgs 30

# CUDA_VISIBLE_DEVICES=2 python ace_run.py --num_parallel_workers 8 --target_class melanoma --source_dir SOURCE_DIR --working_dir SAVE_DIR_Resnet --model_to_run Resnet50 --model_path ../DL-Models-ISIC/graphs/rn50_i18t3nf.pb --labels_path ./labels.txt --bottlenecks LAYER_334 --num_random_exp 40 --max_imgs 50 --min_imgs 30

# CUDA_VISIBLE_DEVICES=2 python ace_run.py --num_parallel_workers 6 --target_class melanoma --source_dir SOURCE_DIR --working_dir Iv4_334_15cluster --model_to_run InceptionV4 --model_path ../DL-Models-ISIC/graphs/iv4_i18t3nf.pb --labels_path ./labels.txt --bottlenecks LAYER_334 --num_random_exp 40 --max_imgs 50 --min_imgs 30

# CUDA_VISIBLE_DEVICES=2 python ace_run.py --num_parallel_workers 6 --target_class melanoma --source_dir SOURCE_DIR --working_dir results_iv4_25c --model_to_run InceptionV4 --model_path ../DL-Models-ISIC/graphs/iv4_i18t3nf.pb --labels_path ./labels.txt --bottlenecks LAYER_334 --num_random_exp 40 --max_imgs 50 --min_imgs 30

# CUDA_VISIBLE_DEVICES=2 python ace_run.py --num_parallel_workers 8 --target_class melanoma --source_dir SOURCE_DIR --working_dir SAVE_DIR_Resnet --model_to_run Resnet50 --model_path ../DL-Models-ISIC/graphs/rn50_i18t3nf.pb --labels_path ./labels.txt --bottlenecks LAYER_334 --num_random_exp 40 --max_imgs 50 --min_imgs 30

# CUDA_VISIBLE_DEVICES=2 python ace_run.py --target_class melanoma --source_dir SOURCE_DIR --working_dir results_rn50_25c_f_seed9f1_newmelimgs --model_to_run Resnet50 --model_path ../DL-Models-ISIC/graphs/rn50_i18t3nf.pb --labels_path ./labels.txt --bottlenecks LAYER_118 --num_random_exp 40 --max_imgs 50 --min_imgs 30