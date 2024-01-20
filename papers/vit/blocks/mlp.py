"""
MLP (or Multilayer perceptron) - A MLP can often refer to any collection of feedforward layers (or in PyTorch's case,
a collection of layers with a forward() method). In the ViT Paper, the authors refer to the MLP as "MLP blocks" and it
contains two torch.nn.Linear() layers with a torch.nn.GELU() non-linearity activation in between them (section 3.1) and
a torch.nn.Dropout() layers after each (Appendex B.1).
"""
