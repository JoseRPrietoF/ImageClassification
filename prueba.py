from transformers import ResNetModel, SwinForImageClassification, ConvNextForImageClassification
import torchvision.models as models

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params+=params
    print(f"Total Trainable Params: {total_params}")
    return total_params
    


if __name__ == "__main__":
    model1 = models.resnet50(pretrained=True)
    print(model1)
    model2 = ResNetModel.from_pretrained("microsoft/resnet-50")
    print(model2)
    print(f'Params for model torchvision {count_parameters(model1)}')
    print(f'Params for model hugginface {count_parameters(model2)}')