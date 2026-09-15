
def set_model_param(model, finetuning_args):
    expert_config = finetuning_args.expert_config
    expert_layer_num=26
    for name, param in model.named_parameters():
        if finetuning_args.gating_layer != None and name == 'model.layers.{}.mlp.gate.weight'.format(finetuning_args.gating_layer):
            print('==========================LOADING layer', 'model.layers.{}.mlp.gate.weight'.format(finetuning_args.gating_layer), ' for update======================================')
            param.requires_grad = True
            continue

        layer_idx, expert_idx = get_idx_deepseek(name)
        if layer_idx == None or str(layer_idx) not in expert_config.keys():
            param.requires_grad = False
            continue
        if layer_idx > expert_layer_num:
            param.requires_grad = False
            continue
        if expert_idx in expert_config[str(layer_idx)]:
            param.requires_grad = True
            print('layer_idx, expert_idx', layer_idx, expert_idx)
        else:
            param.requires_grad = False
    return model


def get_idx_deepseek(layer_name):

    name_list=layer_name.split('.')
    if len(name_list)<4:
        return None, None

    if name_list[1]=='layers' and name_list[4]=='experts':
        return int(name_list[2]), int(name_list[5])
    else:
        return None, None
        
def get_unlearned_expert_config(finetuning_args):

    finetuning_args.gating_layer=None
    if finetuning_args.expert_training_mode == 98:  # DeepSeek WMDP dataset

        expert_config={"16": [20]} # layer start from 1
        finetuning_args.gating_layer=16
    elif finetuning_args.expert_training_mode == 984:  # DeepSeek WMDP, paper-style layer-16 top-4 experts

        expert_config={"16": [20, 10, 15, 54]}
        finetuning_args.gating_layer=16
    elif finetuning_args.expert_training_mode == 988:  # DeepSeek WMDP, paper-style layer-16 top-8 experts

        expert_config={"16": [20, 10, 15, 54, 32, 5, 53, 22]}
        finetuning_args.gating_layer=16
  

    elif finetuning_args.expert_training_mode == 804 : # qwen WMDP dataset    
        expert_config={"4": [5]} # key should be -1 as layer start from 0
        finetuning_args.gating_layer = 4
    else:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Incorrect expert configs mode (corrent mode {})".format(finetuning_args.expert_training_mode))
        exit(1)
    finetuning_args.expert_config=expert_config
    return finetuning_args
