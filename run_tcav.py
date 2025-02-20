import cav as cav
import model as model
import tcav as tcav
import utils as utils
import utils_plot as utils_plot
import os
import torch
import activation_generator as act_gen
import tensorflow as tf
import argparse
from copy import deepcopy
import yaml

def get_tcav_for_target(args):
    model_name = args.model.lower()
    if model_name == "resnet50":
        loader_func = model.ResNet50Wrapper
    elif model_name == "inceptionv3":
        loader_func = model.InceptionV3Wrapper
    elif model_name == "resnet152":
        loader_func = model.ResNet152Wrapper
    elif model_name == "densenet121":
        loader_func = model.DenseNet121Wrapper
    else:
        raise ValueError(f"Model {args.model} not supported sikterlan")
    mymodel = loader_func(args.label_path, args.model_weights)
    act_generator = act_gen.ImageActivationGenerator(mymodel, args.source_dir, args.activation_dir, max_examples=args.max_examples)

    tf.compat.v1.logging.set_verbosity(0)
    
    mytcav = tcav.TCAV(args.target,
                    args.concepts,
                    args.bottlenecks,
                    act_generator,
                    args.alphas,
                    cav_dir=args.cav_dir,
                    random_concepts=args.random_concepts)

    results = mytcav.run()
    print(results)
    
    utils_plot.plot_results(results, num_random_exp=args.num_random_exp)

def main(args):
    original_args = deepcopy(args)
    for target in original_args.targets:
        args = deepcopy(original_args)
        args.working_dir = f'./tcav_class_test/class_{target}'
        args.cav_dir = args.working_dir + '/cavs/'
        args.activation_dir = args.working_dir + '/activations/'
        args.target = target
        utils.make_dir_if_not_exists(args.activation_dir)
        utils.make_dir_if_not_exists(args.working_dir)
        utils.make_dir_if_not_exists(args.cav_dir)
        get_tcav_for_target(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--working_dir', type=str, default='./tcav_class_test')
    parser.add_argument('--source_dir', type=str, default='source_dir')
    parser.add_argument('--targets', type=list, default=[2,3])
    parser.add_argument('--concepts', type=list, default=["Pos-EX_1"])
    parser.add_argument('--random_concepts', type=list, default=[f"Neg-EX_{i}" for i in range(1, 5)])
    parser.add_argument('--num_random_exp', type=int, default=15)
    parser.add_argument('--label_path', type=str, default='labels.txt')
    parser.add_argument('--bottlenecks', type=list, default=['layer3', 'layer4'])
    parser.add_argument('--alphas', type=list, default=[0.1])
    parser.add_argument('--max_examples', type=int, default=100)
    parser.add_argument('--model_weights', type=str, default='weights/best_resnet50.pth')
    parser.add_argument('--config', type=str, default = None)
    parser.add_argument('--model', type=str, default='resnet50')
    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            for key in config:
                setattr(args, key, config[key])
    main(args)
