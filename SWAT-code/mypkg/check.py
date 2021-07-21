import torch
import mypkg
import torch.nn.functional as F
import torch.nn.grad as G

input = torch.randn([1, 1, 4, 4]).cuda()
weight = torch.randn([1, 1, 2, 2]).cuda()
bias = torch.randn([1]).cuda()
stride = [1, 1]
padding = [0, 0]
dilation = [1, 1]
groups = 1

print("#############CONVOLUTION 2 DIMENSION#################")
myresult = mypkg.myConv2d(input, weight, bias, stride, padding, dilation, groups)
actualresult = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
print(f"Are they the same? {torch.equal(myresult, actualresult)}")

print("#############CONVOLUTION BACKWARD#################")
grad_output = torch.randn([1, 1, 3, 3]).cuda()
grad_input, grad_weight = mypkg.myConvBackward(
    input, grad_output, weight, stride, padding, dilation, groups
)
## NOTE CHECK FOR STRIDE PADDING DILATION GROUPS i.e. order of argument
actual_grad_input = G.conv2d_input(
    input.shape, weight, grad_output, stride, padding, dilation, groups
)
actual_grad_weight = G.conv2d_weight(
    input, weight.shape, grad_output, stride, padding, dilation, groups
)
print(f"Are grad_input the same? {torch.allclose(grad_input, actual_grad_input)}")
print(f"Are grad_weight the same? {torch.allclose(grad_weight, actual_grad_weight)}")

print("#############CONVOLUTION BACKWARD INDIVIDUAL ################")
grad_input = mypkg.myConvBackwardInput(
    input.shape, grad_output, weight, stride, padding, dilation, groups
)
grad_weight = mypkg.myConvBackwardWeight(
    weight.shape, grad_output, input, stride, padding, dilation, groups
)
# grad_bias2=mypkg.myConvBackwardBias(grad_output);
# print(grad_bias2)  # NOTE BIAS WILL BE CHECKED WITH PREVIOUS BIAS

# NOTE CHECK FOR STRIDE PADDING DILATION GROUPS i.e. order of argument
actual_grad_input = G.conv2d_input(
    input.shape, weight, grad_output, stride, padding, dilation, groups
)
actual_grad_weight = G.conv2d_weight(
    input, weight.shape, grad_output, stride, padding, dilation, groups
)
print(f"Are grad_input the same? {torch.allclose(grad_input, actual_grad_input)}")
print(f"Are grad_weight the same? {torch.allclose(grad_weight, actual_grad_weight)}")
# print("grad_bias equal\n", grad_bias2 == grad_bias)
