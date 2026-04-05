import argparse

def parse_opts():
    parser = argparse.ArgumentParser(description='options and parameters')

    parser.add_argument(
        '--backbone',
        default='VideoLLaMA2.1-7B-16F',
        type=str,
        help='backbone')

    parser.add_argument(
        '--result_path',
        default='./results',
        type=str,
        help='Result directory path')
    
    parser.add_argument(
        '--val_log_path',
        default='./val_logs',
        type=str,
        help='')
    parser.add_argument(
        '--model_name',
        default='test',
        type=str,
        help='')
    parser.add_argument(
        '--instruct',
        default='',
        type=str,
        help='')

    parser.add_argument(
        '--dataset',
        default='',
        type=str,
        help='')
    parser.add_argument(
        '--phase',
        default='test',
        type=str,
        help='val/test')
    
    parser.add_argument(
        '--resume_path',
        default='',
        type=str,
        help='Save data (.pth) of previous training')
    parser.add_argument(
        '--original_bin',
        default='',
        type=str,
        help='Save data (.pth) of previous training')

    parser.add_argument(
        '--image_path_MINE',
        default='../database/image/',
        type=str,
        help='Directory path of my Videos')

    parser.add_argument(
        '--salmap_path_MINE',
        default='../database/sal_map/',
        type=str,
        help='Salmaps annotations SALICON')

   
    parser.add_argument(
        '--update_step',
        default=16,
        type=int,
        help='parameters update step in training')
    parser.add_argument(
        '--train_step',
        default=16,
        type=int,
        help='sample step in training')
    parser.add_argument(
        '--val_step',
        default=3,
        type=int,
        help='sample step in validation')
    parser.add_argument(
        '--hidden_select_layer',
        default=-1,
        type=int,
        help='hidden select layer of LLM decoder')
    parser.add_argument(
        '--sample_num',
        default=16,
        type=int,
        help='Temporal duration of inputs')
    parser.add_argument(
        '--sample_step',
        default=8,
        type=int,
        help='Temporal step of inputs')
    parser.add_argument(
        '--lr_decoder',
        default=0.01,
        type=float,
        help=
        'Initial learning rate for decoder')
    parser.add_argument(
        '--lr_projector',
        default=0.01,
        type=float,
        help=
        'Initial learning rate for projection')
    parser.add_argument(
        '--lr_lora',
        default=0.1,
        type=float,
        help=
        'Initial learning rate for projection')
    parser.add_argument(
        '--b',
        default=0,
        type=float,
        help=
        'Initial learning rate for projection')
    
    parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum')
    parser.set_defaults(nesterov=False)
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--dampening', default=0.9, type=float, help='dampening of SGD')
    parser.add_argument('--weight_decay', default=0.00001, type=float, help='Weight Decay')

    parser.add_argument(
        '--batch_size',
        default=128, type=int,
        help='Batch size dependent on GPUs memory')
    parser.add_argument(
        '--n_epochs',
        default=0,
        type=int,
        help='Number of total epochs to run')
    parser.add_argument(
        '--begin_epoch',
        default=1,
        type=int,
        help=
        'Training begins at this epoch. Previous trained model indicated by resume_path is loaded.'
    )
    parser.add_argument(
        '--amp',
        action='store_true',
        help='If true, training is not performed.')
    parser.set_defaults(amp=False)
    parser.add_argument(
        '--projection_frozen',
        action='store_true',
        help='If true, train the whole mm_projector.')
    parser.set_defaults(projection_frozen=False)
    parser.add_argument(
        '--mm_train',
        action='store_true',
        help='If true, train the whole mm_projector.')
    parser.set_defaults(mm_train=False)
    parser.add_argument(
        '--precision',
        default='f32',
        type=str,
        help='precision(f32/bf16/f16)')
    parser.add_argument(
        '--finetune',
        action='store_true',
        help='If true, finetune is performed.')
    parser.set_defaults(finetune=False)
    parser.add_argument(
        '--no_train',
        action='store_true',
        help='If true, training is not performed.')
    parser.set_defaults(no_train=False)
    parser.add_argument(
        '--no_val',
        action='store_true',
        help='If true, validation is not performed.')
    parser.set_defaults(no_val=False)
    parser.add_argument(
        '--output_large',
        action='store_true',
        help='If true, validation is not performed.')
    parser.set_defaults(no_val=False)
    parser.add_argument(
        '--test', action='store_true', help='If true, test is performed.')
    parser.set_defaults(test=False)

    parser.add_argument(
        '--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    parser.add_argument(
        '--n_threads',
        default=4,
        type=int,
        help='Number of threads for multi-thread loading')
    parser.add_argument(
        '--checkpoint',
        default=10,
        type=int,
        help='Trained model is saved at every this epochs.')

    parser.add_argument(
        '--lora_r',
        default=128,
        type=int,
        help=
        'lora_rank')
    
    parser.add_argument(
        '--lora_alpha',
        default=256,
        type=int,
        help=
        'lora_alpha')

    parser.add_argument(
        '--lora_dropout',
        default=0.05,
        type=float,
        help=
        'lora_dropout')

    parser.add_argument(
        '--no_vision_tune',
        action='store_true',
        help='If False, tune the vision tower with lora.')
    parser.set_defaults(no_vision_tune=False)

    parser.add_argument(
        '--manual_seed', default=42, type=int, help='Manually set random seed')

    args = parser.parse_args()

    return args
