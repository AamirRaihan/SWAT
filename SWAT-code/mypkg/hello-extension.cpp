#include <torch/extension.h>
#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Exceptions.h>
#include <ATen/cudnn/Handles.h>
#include <ATen/cudnn/Types.h>
#include <ATen/cudnn/Utils.h>
#include <ATen/Functions.h>
#include <ATen/ArrayRef.h>
#include <iostream>
#include <cudnn.h>
#include <vector>
#include <pybind11/pybind11.h>

using CheckedFrom = const char *;
#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS)                      \
    {                                                        \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }
//CHECK FOR REFERENCE i.e. avoid unnecessary call
/*
 * This examples shows how to call cudnn library from pytorch c++ interface 
 */

void AddExamples(
    at::Tensor &input1,
    at::Tensor &input2)
{
  //at::native::Constant one){

  auto handle = at::native::getCudnnHandle();
  //auto dataType = at::native::getCudnnDataType(input1);
  auto dataType = at::native::getCudnnDataTypeFromScalarType(input1.scalar_type());
  at::native::Constant one(dataType, 1);
  at::native::TensorDescriptor idesc, odesc;
  idesc.set(input1);
  odesc.set(input2);
  checkCUDNN(cudnnAddTensor(handle, &one, idesc.desc(), (&input1)->data_ptr(),
                            &one, odesc.desc(), (&input2)->data_ptr()));
}

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
//CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

void cudnnAdd(
    at::Tensor &input1,
    at::Tensor &input2)
{

  CHECK_INPUT(input1);
  CHECK_INPUT(input2);

  AddExamples(input1, input2);
}
at::Tensor myConv2d(
    at::Tensor &input,
    at::Tensor &weight,
    at::Tensor &bias,
    std::vector<int64_t> cstride,
    std::vector<int64_t> cpadding,
    std::vector<int64_t> cdilation,
    int64_t groups)
{
  CHECK_INPUT(input);
  CHECK_INPUT(weight);
  CHECK_INPUT(bias);
  at::IntArrayRef stride{cstride};
  at::IntArrayRef padding{cpadding};
  at::IntArrayRef dilation{cdilation};

  //torch::Tensor ginput  = input.to(torch::kCUDA);
  //torch::Tensor gweight = weight.to(torch::kCUDA);
  //torch::Tensor gbias   = bias.to(torch::kCUDA);
  //auto output = at::empty({0},input.options());
  return (at::conv2d(input, weight, bias, stride, padding, dilation, groups));
}

//std::tuple<at::Tensor, at::Tensor, at::Tensor> mycudnn_convolution_backward(
std::tuple<at::Tensor, at::Tensor> mycudnn_convolution_backward(
    const at::Tensor &input,
    const at::Tensor &grad_output_t,
    const at::Tensor &weight,
    std::vector<int64_t> cstride,
    std::vector<int64_t> cpadding,
    std::vector<int64_t> cdilation,
    int64_t groups)
{
  bool allow_tf32 = true; //https://pytorch.org/docs/stable/backends.html
  bool benchmark = false;
  bool deterministic = true;
  //std::array<bool, 3> output_mask{true, true, true};
  std::array<bool, 2> output_mask{true, true};
  CHECK_INPUT(input);
  CHECK_INPUT(weight);
  CHECK_INPUT(grad_output_t);
  at::IntArrayRef stride{cstride};
  at::IntArrayRef padding{cpadding};
  at::IntArrayRef dilation{cdilation};

  //return (cudnn_convolution_backward(input, grad_output_t, weight, padding, stride, dilation, groups, benchmark, deterministic, output_mask));
  // Might need to include bias here
  return (cudnn_convolution_backward(input, grad_output_t, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32, output_mask));
}

at::Tensor mycudnn_convolution_backward_input(
    std::vector<int64_t> cinput_size,
    const at::Tensor &grad_output,
    const at::Tensor &weight,
    std::vector<int64_t> cstride,
    std::vector<int64_t> cpadding,
    std::vector<int64_t> cdilation,
    int64_t groups)
{
  bool allow_tf32 = true;
  bool benchmark = false;
  bool deterministic = true;

  CHECK_INPUT(weight);
  CHECK_INPUT(grad_output);
  at::IntArrayRef stride{cstride};
  at::IntArrayRef padding{cpadding};
  at::IntArrayRef dilation{cdilation};
  at::IntArrayRef input_size{cinput_size};

  return (cudnn_convolution_backward_input(input_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32));
}
at::Tensor mycudnn_convolution_backward_weight(
    std::vector<int64_t> cweight_size,
    const at::Tensor &grad_output,
    const at::Tensor &input,
    std::vector<int64_t> cstride,
    std::vector<int64_t> cpadding,
    std::vector<int64_t> cdilation,
    int64_t groups)
{
  bool allow_tf32 = true;
  bool benchmark = false;
  bool deterministic = true;

  CHECK_INPUT(grad_output);
  CHECK_INPUT(input);
  at::IntArrayRef stride{cstride};
  at::IntArrayRef padding{cpadding};
  at::IntArrayRef dilation{cdilation};
  at::IntArrayRef weight_size{cweight_size};

  return (cudnn_convolution_backward_weight(weight_size, grad_output, input, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32));
}

// Now outside convolution. cudnn_convolution_backward_bias no longer exists
//at::Tensor mycudnn_convolution_backward_bias(const at::Tensor &grad_output)
//{
//  CHECK_INPUT(grad_output);
//  return (cudnn_convolution_backward_bias(grad_output));
//}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> mycudnn_batch_norm(
    const at::Tensor &input_t, const at::Tensor &weight_t,
    const at::Tensor &bias_t, const at::Tensor &running_mean_t, const at::Tensor &running_var_t,
    bool training, double exponential_average_factor, double epsilon)
{

  auto out = torch::cudnn_batch_norm(input_t, weight_t, bias_t, running_mean_t, running_var_t, training, exponential_average_factor, epsilon);

  if (training == true)
  {
    return out;
  }
  else
  {
    int64_t n_input = input_t.size(1);
    at::Tensor save_mean = at::empty({n_input}, input_t.options());
    at::Tensor save_var = at::empty({n_input}, input_t.options());

    at::TensorArg input{input_t, "input", 1};
    //at::Tensor reserve = at::empty({0}, input->options().dtype(c10::kByte)); // Added from https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cudnn/BatchNorm.cpp#L214
    at::Tensor reserve = at::empty({0}, input_t.options()); // Added from https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cudnn/BatchNorm.cpp#L214
    auto val = std::get<0>(out);

    return std::make_tuple(val, save_mean, save_var, reserve); // Added reserve
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> mycudnn_batch_norm_backward(
    const at::Tensor &input_t, const at::Tensor &grad_output_t, const at::Tensor &weight_t,
    // Unused: but we require them to be passed so that double backwards
    // has access
    const at::Tensor &running_mean, const at::Tensor &running_var,
    const at::Tensor &save_mean_t, const at::Tensor &save_var_t,
    double epsilon, const at::Tensor &reserveSpace)
{

  return torch::native::cudnn_batch_norm_backward(input_t, grad_output_t, weight_t, running_mean, running_var, save_mean_t, save_var_t, epsilon, reserveSpace);
}

void mycheck(at::Tensor input, at::Tensor weight, at::Tensor bias, std::vector<int64_t> stride)
{
  std::cout << "Called Successfully" << std::endl;
  for (unsigned int i = 0; i < stride.size(); i++)
  {
    std::cout << stride[i];
  }
  std::cout << input << std::endl;
  std::cout << weight << std::endl;
  std::cout << bias << std::endl;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("cudnnAdd", &cudnnAdd, "(cuDNN ADD)");
  m.def("myConv2d", &myConv2d, "(conv2d)");
  m.def("myConvBackward", &mycudnn_convolution_backward, "grad output=<input,weight,bias>");
  m.def("myConvBackwardInput", &mycudnn_convolution_backward_input, "grad output=<input>");
  m.def("myConvBackwardWeight", &mycudnn_convolution_backward_weight, "grad output=<weight>");
  //m.def("myConvBackwardBias", &mycudnn_convolution_backward_bias, "grad output=<bias>");
  m.def("myBatchNormBackward", &mycudnn_batch_norm_backward, "grad output=<grad_input,grad_weight,grad_bias>");
  m.def("myBatchNormForward", &mycudnn_batch_norm, "output=<normalized,savemean,savevar>");
  m.def("mycheck", &mycheck, "(check)");
}
