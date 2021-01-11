import torch
import torch.nn as nn
import numpy as np
import math
import custom_layers.custom_conv as C
import custom_layers.custom_linear as L
import custom_layers.custom_batchnorm as B 
import pdb
import numpy as np
import topk
# This file is adapted from the Sparse Network From Scratch codebase (https://github.com/TimDettmers/sparse_learning)  and 
# Rigging the Lottery Ticket codebase (https://github.com/google-research/rigl).
class layerSparsity(object):
    def __init__(self,optimizer,pruning_type,sparsify_first_layer,sparsify_last_layer, inference=False):
        self.pruning_type=pruning_type
        self.sparsify_first_layer=sparsify_first_layer
        self.sparsify_last_layer=sparsify_last_layer
        self.verbose=True
        self.optimizer=optimizer
        self.modules=[]
        self.mode=None
        self.num_layer=0

        self.layers={}
        self.sparsity={}
        self.layer_momentum = {}
        self.layer_non_zeros = {}
        self.layer_zeros = {}
        self.mask={}
        self.total_density=0
        self.total_zero = 0
        self.total_non_zero = 0
        self.total_parameter=0
        self.total_momentum = 0.0
        self.inference=inference
            
    def get_sparsity(self):
        return self.sparsity

    def add_module(self, module, density, mode='constant'):
        self.modules.append(module)
        self.total_density=density
        self.mode=mode

        for name, tensor in module.named_parameters():
            self.layers[name]=tensor.shape
        
        print('Removing biases...')
        self.remove_weight_partial_name('bias')
        print('Removing 2D batch norms...')
        self.remove_type(nn.BatchNorm2d, verbose=self.verbose)
        self.remove_type(B.CustomBatchNorm2d, verbose=self.verbose)
        print('Removing 1D batch norms...')
        self.remove_type(nn.BatchNorm1d, verbose=self.verbose)
        
        layer_num=0
        conv_linear={}
        for name, tensor in module.named_parameters():
            if name not in self.layers: continue
            self.layers[name]=(tensor.shape,layer_num)
            self.total_parameter+=np.prod(tensor.shape)
            print(name,":",layer_num)
            conv_linear[name]=layer_num
            layer_num+=1
        torch.save(conv_linear,'./conv_linear_name')
        self.num_layer=layer_num

        #     Assumption: model is already pruned model for inference. 
        #   After training with SWAT-U/ERK/M  we pruned each layer of the 
        # network to the desired sparsity learned for that network architecture. 
        if self.inference:
            self.init(mode='constant', density=1)
            return

        if mode in ['constant','momentum']:
            self.init(mode='constant', density=density)
        if 'erdos_renyi' in mode:
            self.init(mode='constant', density=density)
            if 'kernel' in mode:
                print("MODE:ERDOS_RENYI_KERNEL")
                self.erdos_renyi_kernel(include_kernel=True)
            else:
                print("MODE:ERDOS_RENYI")
                self.erdos_renyi_kernel(include_kernel=False)
            self.print_debug()
        else:
            self.init(mode=mode, density=density)
    
    def sparsity_regrowth(self):
        num_overgrowth = 0
        total_overgrowth = 0
        residual = 0

        residual = 9999
        mean_residual = 0
        freeze_layer = {}
        i = 0
        total_nonzero=0
        target_nonzero=self.total_density*self.total_parameter
        while residual > 0 and i < 1000:
            if total_nonzero > target_nonzero:
                break
            residual = 0
            total_nonzero=0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.layers: continue
                    #print(name,weight.shape)
                    if name in freeze_layer:
                        regrowth = freeze_layer[name]
                    else:
                        regrowth =math.ceil(self.layer_momentum[name]*self.total_density*self.total_parameter)
                    regrowth += mean_residual
                    max_regrowth=np.prod(self.layers[name][0])
                    if regrowth > 0.99*max_regrowth:
                        freeze_layer[name] = max_regrowth
                        residual += regrowth - freeze_layer[name]
                    else:
                        freeze_layer[name] = regrowth
                    total_nonzero+=freeze_layer[name]
            if len(freeze_layer) == 0: mean_residual = 0
            else:
                if residual > len(freeze_layer):
                    mean_residual = math.ceil(residual / len(freeze_layer))
                else:
                    residual = 0
            i += 1

        if i == 1000:
            print('Error resolving the residual! Layers are too full! Residual left over: {0}'.format(residual))
        
        #for name,item in freeze_layer.items():
        #    print(name," : ",item)
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.layers: continue
                #print("SP:",name,":",1-freeze_layer[name]/np.prod(self.layers[name][0]))
                self.sparsity[self.layers[name][1]] = (name,1-freeze_layer[name]/np.prod(self.layers[name][0]))
        return 
    def gather_statistic(self): 
        if self.mode=='constant' or self.mode=='erdos_renyi_kernel' or self.mode=='erdos_renyi':
            #self.print_debug()
            return
        else:
            #RECOMPUTING
            self.layer_non_zeros = {}
            self.layer_zeros = {}
            self.total_non_zero = 0
            self.total_momentum = 0.0
            gpuid=torch.cuda.current_device()
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.layers: continue
                    if self.pruning_type=='unstructured':
                        wt_threshold=torch.load('./dump/threshold_'+str(int(self.layers[name][1]))+'_'+str(gpuid))[0]
                        self.mask[name]=weight.abs()>=(wt_threshold.cuda()).type_as(weight)
                    else:
                        if len(weight.shape)==4:
                            if self.pruning_type=='structured_channel':
                                self.mask[name]=topk.drop_structured(weight,1-(self.sparsity[self.layers[name][1]][1]))!=0
                            else:
                                self.mask[name]=topk.drop_structured_filter(weight,1-(self.sparsity[self.layers[name][1]][1]))!=0
                        else:
                            self.mask[name]=topk.matrix_drop(weight,1-(self.sparsity[self.layers[name][1]][1]))!=0
                    mask=self.mask[name]
                    self.layer_non_zeros[name] = mask.sum().item()
                    self.layer_zeros[name] = mask.numel() - self.layer_non_zeros[name]
                    self.total_non_zero += self.layer_non_zeros[name]
                    self.total_zero += self.layer_zeros[name]
        

    def erdos_renyi_kernel(self,include_kernel=True):
        is_eps_valid = False
        # # The following loop will terminate worst case when all masks are in the
        # custom_sparsity_map. This should probably never happen though, since once
        # we have a single variable or more with the same constant, we have a valid
        # epsilon. Note that for each iteration we add at least one variable to the
        # custom_sparsity_map and therefore this while loop should terminate.
        dense_layers = set()
        while not is_eps_valid:
            # We will start with all layers and try to find right epsilon. However if
            # any probablity exceeds 1, we will make that layer dense and repeat the
            # process (finding epsilon) with the non-dense layers.
            # We want the total number of connections to be the same. Let say we have
            # for layers with N_1, ..., N_4 parameters each. Let say after some
            # iterations probability of some dense layers (3, 4) exceeded 1 and
            # therefore we added them to the dense_layers set. Those layers will not
            # scale with erdos_renyi, however we need to count them so that target
            # paratemeter count is achieved. See below.
            # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
            #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
            # eps * (p_1 * N_1 + p_2 * N_2) =
            #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
            # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.

            divisor = 0
            rhs = 0
            raw_probabilities = {}
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.layers: continue
                    var_name=name
                    shape_list=self.layers[name][0]
                    n_param=np.prod(shape_list)
                    n_zeros=self.layer_zeros[name]
                    if var_name in dense_layers:
                        # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
                        rhs -= n_zeros
                    #elif var_name in custom_sparsity_map:
                    #    # We ignore custom_sparsities in erdos-renyi calculations.
                    #    pass
                    else:
                        # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
                        # equation above.
                        n_ones = n_param - n_zeros
                        rhs += n_ones
                     
                        # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
                        if include_kernel:
                            raw_probabilities[name] = (np.sum(shape_list) /
                                                              np.prod(shape_list))
                        else:
                            n_in, n_out = shape_list[-2:]
                            raw_probabilities[name] = (n_in + n_out) / (n_in * n_out)
                        # Note that raw_probabilities[mask] * n_param gives the individual
                        # elements of the divisor.
                        divisor += raw_probabilities[name] * n_param
            # By multipliying individual probabilites with epsilon, we should get the
            # number of parameters per layer correctly.
            eps = rhs / divisor
            # If eps * raw_probabilities[mask.name] > 1. We set the sparsities of that
            # mask to 0., so they become part of dense_layers sets.
            max_prob = np.max(list(raw_probabilities.values()))
            max_prob_one = max_prob * eps
            if max_prob_one > 1:
                is_eps_valid = False
                for mask_name, mask_raw_prob in raw_probabilities.items():
                    if mask_raw_prob == max_prob:
                        print(" Sparsity of var: {} had to be set to 0.".format(mask_name))
                        dense_layers.add(mask_name)
            else:
                is_eps_valid = True

        sparsities = {}
        # With the valid epsilon, we can set sparsities of the remaning layers.
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.layers: continue
                shape_list=self.layers[name][0]
                n_param = np.prod(shape_list)
                if name in dense_layers:
                  sparsities[name] = 0.
                else:
                  probability_one = eps * raw_probabilities[name]
                  sparsities[name] = 1. - probability_one
                  #print("layer: {}, shape: {}, sparsity: {}".format(name, list(shape_list), sparsities[name]))
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.layers: continue
                #@print("SP:",name,":",1-freeze_layer[name]/np.prod(self.layers[name][0]))
                self.sparsity[self.layers[name][1]] = (name,sparsities[name])
        return 

    def step(self):
        if self.mode=='constant' or self.mode=='erdos_renyi_kernel' or self.mode=='erdos_renyi':
            #self.print_debug()
            return
        
        if self.mode=='momentum':
            #self.print_debug()
            self.gather_momentum()
            self.sparsity_regrowth()        
    def print_debug(self):
        zero=0
        nonzero=0
        total=0
        for layer,values in self.sparsity.items():
            name=values[0]
            sparsity=values[1]
            parameter_in_layer=np.prod(self.layers[name][0])
            print("Layer : ",layer,"(",name,"):"," Sparsity: ",round(sparsity,4)," Size:",parameter_in_layer,round(parameter_in_layer*sparsity))
            nonzero+=round(parameter_in_layer*(1-sparsity))
            zero+=round(parameter_in_layer*sparsity)
            total+=parameter_in_layer
        #    total=0
        #    for name,mom in self.layer_momentum.items():
        #        total+=mom
        #        print("Momentum: ",name," : ",mom)
        #    print("TOTAL_MOMENTUM",total)
        print("TOTAL_NON_ZEROS",self.total_non_zero,nonzero)
        print("TOTAL_ZEROS",self.total_zero,zero)
        print("TOTAL_PARAMETER",self.total_parameter,total)
        print("FOUND SPARSITY:",1-self.total_non_zero/self.total_parameter,zero/total)


    def get_momentum_for_weight(self, weight):
        if 'exp_avg' in self.optimizer.state[weight]:
            adam_m1 = self.optimizer.state[weight]['exp_avg']
            adam_m2 = self.optimizer.state[weight]['exp_avg_sq']
            grad = adam_m1/(torch.sqrt(adam_m2) + 1e-08)
        elif 'momentum_buffer' in self.optimizer.state[weight]:
            grad = self.optimizer.state[weight]['momentum_buffer']

        return grad

    def momentum_redistribution(self, weight, mask):
        grad = self.get_momentum_for_weight(weight)
        mean_magnitude = torch.abs(grad[mask]).mean().item()
        return mean_magnitude

    def gather_momentum(self):
        gpuid=torch.cuda.current_device()
        self.layer_momentum = {}
        self.total_momentum = 0.0
        
        
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.layers: continue
                mask                      = self.mask[name]
                self.layer_momentum[name] = self.momentum_redistribution(weight,mask)

                if not np.isnan(self.layer_momentum[name]):
                    self.total_momentum += self.layer_momentum[name]

        for name in self.layer_momentum:
            if self.total_momentum != 0.0:
                self.layer_momentum[name] /= self.total_momentum
            else:
                print('Total Momentum was zero!')
                exit()
            #print("Layer:",name, "Momentum: ",self.layer_momentum[name])

    def init(self, mode='constant', density=1.0):
        if mode == 'constant':
            # initializes each layer with a constant percentage of dense weights
            # each layer will have weight.numel()*density weights.
            # weight.numel()*density == weight.numel()*(1.0-sparsity)
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.layers: continue
                    if (not self.sparsify_first_layer) and self.layers[name][1]==0:
                        self.layer_non_zeros[name]=weight.numel()
                        sparsity=0
                    elif (not self.sparsify_last_layer) and self.layers[name][1]==self.num_layer-1:
                        self.layer_non_zeros[name]=weight.numel()
                        sparsity=0
                    else:
                        self.layer_non_zeros[name]=weight.numel()*density
                        sparsity=1-density
                    self.layer_zeros[name] = weight.numel() - self.layer_non_zeros[name]
                    self.sparsity[self.layers[name][1]]=(name,sparsity)
                    self.total_non_zero += self.layer_non_zeros[name]
                    self.total_zero += self.layer_zeros[name]
            
            self.total_momentum=0

    def remove_weight(self, name):
        if name in self.layers:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.layers[name][0], self.layers[name][0].numel()))
            self.layers.pop(name)
        elif name+'.weight' in self.layers:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.layers[name+'.weight'][0], np.prod(self.layers[name+'.weight'][0])))
            self.layers.pop(name+'.weight')
        else:
            print('ERROR',name)

    def remove_weight_partial_name(self, partial_name, verbose=False):
        removed = set()
        for name in list(self.layers.keys()):
            if partial_name in name:
                if self.verbose:
                    print('Removing {0} of size {1} with {2} parameters...'.format(name, self.layers[name][0], np.prod(self.layers[name][0])))
                removed.add(name)
                self.layers.pop(name)
    
        print('Removed {0} layers.'.format(len(removed)))
    
    
    
    def remove_type(self, nn_type, verbose=False):
        for module in self.modules:
            for name, module in module.named_modules():
                if isinstance(module, nn_type):
                    self.remove_weight(name)
    
