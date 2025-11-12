import os
import torch

# Try to use non-FX quantization API for complex models like GPT-2
try:
    from torch.ao.quantization import prepare, convert

    print('path taken')
    
    def prepare_fx_compat(model: torch.nn.Module, qconfig_mapping, example_inputs):
        """Use non-FX quantization for complex models"""
        import torch.nn as nn
        from torch.ao.quantization import propagate_qconfig_, float_qparams_weight_only_qconfig
        
        model.eval()
        # Convert QConfigMapping to dict for propagate_qconfig_
        qconfig_dict = qconfig_mapping.to_dict()
        propagate_qconfig_(model, qconfig_dict)
        
        # Manually set qconfig on embedding modules to ensure they use float_qparams_weight_only_qconfig
        # This is needed because propagate_qconfig_ might not properly handle object_type mappings
        # If an embedding module has a qconfig set (meaning it's being quantized), we need to ensure
        # it uses the correct float_qparams_weight_only_qconfig, as required by PyTorch
        for name, module in model.named_modules():
            if isinstance(module, nn.Embedding):
                # If the module has a qconfig (is being quantized), ensure it's the correct one
                if hasattr(module, 'qconfig') and module.qconfig is not None:
                    # PyTorch requires float_qparams_weight_only_qconfig for embedding quantization
                    module.qconfig = float_qparams_weight_only_qconfig
        
        # Prepare the model (this doesn't use FX)
        prepared = prepare(model)
        return prepared

    def convert_fx_compat(prepared_model: torch.nn.Module):
        """Convert prepared model to quantized"""
        return convert(prepared_model)

except Exception:
    # Fallback: low-level API; we must trace & provide node_name_to_scope
    from torch.fx import symbolic_trace
    from torch.ao.quantization.fx import prepare as _fx_prepare  # low-level
    from torch.ao.quantization.fx import convert as _fx_convert  # low-level
    # Create a simple node_name_to_scope mapping function since it doesn't exist in this PyTorch version
    def get_node_name_to_scope(gm):
        """Create a simple node_name_to_scope mapping for the graph module"""
        node_name_to_scope = {}
        for node in gm.graph.nodes:
            if node.op == 'call_module':
                # For call_module nodes, use the module name as scope
                module_name = node.target
                node_name_to_scope[node.name] = (module_name, type(gm.get_submodule(module_name)))
            else:
                # For other nodes, use the root scope
                node_name_to_scope[node.name] = ('', type(gm))
        return node_name_to_scope

    def prepare_fx_compat(model: torch.nn.Module, qconfig_mapping, example_inputs):
        model.eval()
        # Try to trace with more concrete args to avoid control flow issues
        try:
            # First try with minimal concrete args
            gm = symbolic_trace(model, concrete_args={"inputs_embeds": None, "attention_mask": None})
        except Exception:
            # If that fails, try with even more concrete args
            position_ids = torch.arange(example_inputs[0].size(1), device=example_inputs[0].device).unsqueeze(0)
            gm = symbolic_trace(model, concrete_args={
                "inputs_embeds": None, 
                "attention_mask": None, 
                "position_ids": position_ids,
                "head_mask": None,
                "encoder_hidden_states": None,
                "encoder_attention_mask": None,
                "past_key_values": None,
                "use_cache": False,
                "output_attentions": False,
                "output_hidden_states": False,
                "return_dict": False
            })
        
        node_map = get_node_name_to_scope(gm)
        return _fx_prepare(gm, qconfig_mapping, False, node_map, example_inputs)

    def convert_fx_compat(prepared_model: torch.nn.Module):
        return _fx_convert(prepared_model)