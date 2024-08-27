float mse_loss(Tensor *input, Tensor *target, Tensor *input_grad);
void forward_conv(Tensor *input, Tensor *conv, Tensor *output);
void backward_conv(Tensor* conv_grad, Tensor *input, Tensor *output_grad);
void backward_conv_input(Tensor *input_grad, Tensor *conv, Tensor *output_grad);
