import torchvision as tv

train = tv.datasets.Places365(
    root = 'data',                       
    download = True,            
)