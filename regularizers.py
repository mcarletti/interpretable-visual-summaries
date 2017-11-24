import torch


def l1_reg(input_tensor):
    '''The input_tensor has values between 0 and 1.
    Encourage to reduce the number of masked pixels:
    0 = totally mask the pixel
    1 = do not mask the pixel
    '''
    return torch.mean(torch.abs(1 - input_tensor))


def tv_reg(input_tensor, tv_beta):
    '''The input_tensor has values between 0 and 1.
    Compute the gradients on rows and cols saparately
    and sum them. This regularizer encourages smooth and
    compact heatmaps, avoiding sparse pixel masking.
    '''
    img = input_tensor[0, 0, :]
    row_grad = torch.mean(torch.abs((img[:-1 , :] - img[1 :, :])).pow(tv_beta))
    col_grad = torch.mean(torch.abs((img[: , :-1] - img[: , 1 :])).pow(tv_beta))
    return row_grad + col_grad


def less_reg(predictions, target_predictions):
    '''Compute the difference between the predictions.
    Encourages lower predictions than the target.
    '''
    return torch.mean(torch.clamp(predictions - target_predictions, min=0))


def lasso_reg(input_tensor):
    '''The input_tensor has values between 0 and 1.
    Force the optimizer to convert the masked pixel values
    to 0 or 1.
    '''
    return torch.mean(torch.abs((1 - input_tensor) * input_tensor))
